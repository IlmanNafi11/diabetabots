"""Telegram message handlers wiring state machine, services, and Telegram API.

This module owns the stateless handler class (:class:`TelegramUpdateHandler`).
It receives a :class:`TelegramUpdate` from the polling loop and dispatches it to
the appropriate method based on conversation state and update type.

Wire-in order (checked at Checkpoint 3):
  1. TelegramUpdateHandler — entry; receives raw update; routes by state.
  2. ConversationManager — in-memory registration state machine.
  3. RegistrationService — user profile upsert (needs DB session).
  4. RAGService — ChromaDB food retrieval.
  5. LLMService — analysis narrative.
  6. EnergyService — daily energy balance (needs DB session).

All user-facing strings are Bahasa Indonesia to match §3.5 PRD.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.bot.conversation import ConversationManager, ConversationState
from app.bot.validators import (
    ValidationError,
    parse_activity,
    parse_age,
    parse_gender,
    parse_gram,
    parse_height,
    parse_weight,
)
from app.services.energy_service import ConfirmationStatus, EnergyService
from app.services.llm_service import LLMService, UserProfile
from app.services.nutrition_service import glycemic_load, nutrition_for_gram
from app.services.rag_service import RAGService
from app.services.registration_service import RegistrationService
from app.services.telegram_service import TelegramService, TelegramUpdate, TelegramUpdateType
from app.services.update_limiter import UpdateLimiter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

# Portions that describe count/volume rather than mass — always clarify.
_NON_MASS_PORTION_RE = re.compile(
    r"\d+\s*(piring|butir|mangkuk|gelas|sendok|buah|potong|cangkir|cup)\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Inline keyboard layouts
# ---------------------------------------------------------------------------

# Callback data carries the pending_analyses row id so a tap resolves against
# the exact verified payload shown, never a re-derivation (PRD section 8.1).
_CONFIRM_YA_PREFIX = "confirm_ya:"
_CONFIRM_TIDAK_PREFIX = "confirm_tidak:"


def build_confirm_keyboard(pending_id: UUID) -> list[list[dict[str, str]]]:
    """Return the "Catat sebagai konsumsi?" inline keyboard for a pending row."""
    return [
        [
            {"text": "Ya", "callback_data": f"{_CONFIRM_YA_PREFIX}{pending_id}"},
            {"text": "Tidak", "callback_data": f"{_CONFIRM_TIDAK_PREFIX}{pending_id}"},
        ]
    ]


# ---------------------------------------------------------------------------
# TelegramUpdateHandler
# ---------------------------------------------------------------------------


class TelegramUpdateHandler:
    """Route Telegram updates to service calls based on conversation state.

    Parameters
    ----------
    telegram_svc:
        Outbound API client (send messages, answer callbacks).
    conversation_manager:
        In-memory state machine (one per handler instance).
    registration_svc:
        Async user-profile upsert (needs a session for DB writes).
    energy_svc:
        Async daily-energy CRUD (needs a session for DB writes).
    rag_svc:
        ChromaDB food retrieval.
    llm_svc:
        LLM food analysis.
    session:
        Optional async DB session for services that need it.
    """

    def __init__(
        self,
        telegram_svc: TelegramService,
        conversation_manager: ConversationManager,
        registration_svc: RegistrationService,
        energy_svc: EnergyService,
        rag_svc: RAGService,
        llm_svc: LLMService,
        session: AsyncSession | None = None,
        max_user_question_chars: int = 512,
        update_limiter: UpdateLimiter | None = None,
    ) -> None:
        self._telegram_svc = telegram_svc
        self._conversation_manager = conversation_manager
        self._registration_svc = registration_svc
        self._energy_svc = energy_svc
        self._rag_svc = rag_svc
        self._llm_svc = llm_svc
        self._session = session
        self._max_user_question_chars = max_user_question_chars
        self._update_limiter = update_limiter
    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_update(self, update: TelegramUpdate) -> None:
        """Route an update and dispatch to the appropriate handler method."""
        if update.update_type == TelegramUpdateType.CALLBACK_QUERY:
            await self._handle_callback(update)
        elif update.update_type == TelegramUpdateType.MESSAGE:
            await self._handle_message(update)
        # OTHER update types (inline_query, etc.) are silently ignored.

    # ------------------------------------------------------------------
    # Message handler (by state)
    # ------------------------------------------------------------------

    async def _handle_message(self, update: TelegramUpdate) -> None:
        """Dispatch a message based on the current conversation state."""
        if update.chat_id is None:
            return

        # /start restarts registration from any non-idle state
        if update.text == "/start":
            self._conversation_manager.start_registration(update.chat_id)
            await self._send_registration_gender_prompt(update.chat_id)
            return
        if update.text == "/hapus_data":
            await self._handle_delete_data(update.chat_id)
            return

        conv = self._conversation_manager.get(update.chat_id)

        match conv.state:
            case ConversationState.IDLE:
                await self._send_welcome_or_prompt_registration(update.chat_id)

            case ConversationState.AWAIT_GENDER:
                await self._handle_await_gender(update.chat_id, update.text or "")

            case ConversationState.AWAIT_AGE:
                await self._handle_await_age(update.chat_id, update.text or "")

            case ConversationState.AWAIT_WEIGHT:
                await self._handle_await_weight(update.chat_id, update.text or "")

            case ConversationState.AWAIT_HEIGHT:
                await self._handle_await_height(update.chat_id, update.text or "")

            case ConversationState.AWAIT_ACTIVITY:
                await self._handle_await_activity(update.chat_id, update.text or "")

            case ConversationState.CONFIRMED:
                await self._handle_confirmed(update.chat_id, update.text or "")

            case ConversationState.AWAIT_GRAM_CLARIFY:
                await self._handle_await_gram_clarify(update.chat_id, update.text or "")

    # ------------------------------------------------------------------
    # Callback query handler
    # ------------------------------------------------------------------

    async def _handle_callback(self, update: TelegramUpdate) -> None:
        """Handle inline button taps bound to a persisted pending row."""
        if update.chat_id is None or update.callback_data is None:
            return

        try:
            action, pending_text = update.callback_data.split(":", 1)
            pending_id = UUID(pending_text)
        except (ValueError, AttributeError):
            await self._send(update.chat_id, "Analisis tidak berlaku.")
            return

        message_id = update.message_id
        if action == "confirm_ya":
            await self._handle_confirm_ya(update.chat_id, pending_id, message_id)
        elif action == "confirm_tidak":
            await self._handle_confirm_tidak(update.chat_id, pending_id, message_id)
        else:
            await self._send(update.chat_id, "Analisis tidak berlaku.")
    # ------------------------------------------------------------------
    # Data deletion (PDP right-to-erasure)
    # ------------------------------------------------------------------

    async def _handle_delete_data(self, chat_id: int) -> None:
        """Delete the user's stored data and reset local conversation state."""
        self._conversation_manager.clear(chat_id)
        try:
            deleted = await self._registration_svc.delete_user_data(chat_id)
        except SQLAlchemyError:
            log.error("Data deletion failed for chat %s", chat_id)
            await self._send(
                chat_id, "Maaf, gagal menghapus data. Silakan coba lagi beberapa saat."
            )
            return
        if deleted:
            await self._send(
                chat_id,
                "✅ Semua data Anda telah dihapus dari sistem kami. "
                "Untuk menggunakan bot lagi, ketik /start.",
            )
        else:
            await self._send(
                chat_id,
                "Kami tidak menemukan data Anda. Ketik /start untuk mulai menggunakan bot.",
            )

    # ------------------------------------------------------------------
    # Registration step handlers
    # ------------------------------------------------------------------

    async def _handle_await_gender(self, chat_id: int, text: str) -> None:
        try:
            gender = parse_gender(text)
            self._conversation_manager.accept_gender(chat_id, gender)
            await self._send_registration_age_prompt(chat_id)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send_registration_gender_prompt(chat_id)

    async def _handle_await_age(self, chat_id: int, text: str) -> None:
        try:
            age = parse_age(text)
            self._conversation_manager.accept_age(chat_id, age)
            await self._send_registration_weight_prompt(chat_id)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send_registration_age_prompt(chat_id)

    async def _handle_await_weight(self, chat_id: int, text: str) -> None:
        try:
            weight = parse_weight(text)
            self._conversation_manager.accept_weight(chat_id, weight)
            await self._send_registration_height_prompt(chat_id)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send_registration_weight_prompt(chat_id)

    async def _handle_await_height(self, chat_id: int, text: str) -> None:
        try:
            height = parse_height(text)
            self._conversation_manager.accept_height(chat_id, height)
            await self._send_registration_activity_prompt(chat_id)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send_registration_height_prompt(chat_id)

    async def _handle_await_activity(self, chat_id: int, text: str) -> None:
        try:
            activity = parse_activity(text)
            self._conversation_manager.accept_activity(chat_id, activity)
            # Persist to DB
            conv = self._conversation_manager.get(chat_id)
            profile = conv.profile
            await self._registration_svc.register_profile(
                telegram_id=chat_id,
                gender=profile.gender,
                age=profile.age,
                weight_kg=profile.weight_kg,
                height_cm=profile.height_cm,
                activity=profile.activity,
            )
            await self._send_registration_confirmed(chat_id)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send_registration_activity_prompt(chat_id)

    # ------------------------------------------------------------------
    # CONFIRMED state — food query
    # ------------------------------------------------------------------

    async def _handle_confirmed(self, chat_id: int, text: str) -> None:
        """Process a food query: parse gram → retrieve → analyze → send with buttons."""
        if not text or not text.strip():
            await self._send(chat_id, "Ketik nama makanan, contoh: nasi goreng 150g")
            return
        if len(text) > self._max_user_question_chars:
            await self._send(
                chat_id,
                f"Pertanyaan terlalu panjang (maksimal "
                f"{self._max_user_question_chars} karakter). Coba ketik lebih singkat.",
            )
            return

        if (
            self._update_limiter is not None
            and not self._update_limiter.allow(chat_id)
        ):
            await self._send(
                chat_id,
                "Mohon tunggu sebentar sebelum mengirim pertanyaan berikutnya.",
            )
            return

        food_query = text.strip()

        # Non-mass portion units ("1 piring", "2 butir") must be clarified, never
        # assumed as grams (SPEC section 7.3).
        if _NON_MASS_PORTION_RE.search(food_query):
            self._conversation_manager.request_gram_clarification(chat_id, food_query)
            await self._send(
                chat_id,
                f"Berapa gram untuk '{food_query}'? "
                "Saya tidak bisa mengasumsikan takaran. Ketik jumlah dalam gram, contoh: 150",
            )
            return

        # Try to split "name gram" — if gram is unparseable, ask for clarification
        gram: float | None = None
        for part in reversed(food_query.split()):
            try:
                gram = parse_gram(part)
                food_query = food_query[: -len(part)].strip()
                break
            except ValidationError:
                pass  # Not a gram — keep trying

        if gram is None or gram <= 0:
            # Cannot determine gram from the input → ask user
            self._conversation_manager.request_gram_clarification(chat_id, food_query)
            await self._send(
                chat_id, f"Berapa gram untuk '{food_query}'? Ketik jumlah dalam gram, contoh: 150"
            )
            return

        await self._analyze_and_send(chat_id, food_query, gram)

    async def _handle_await_gram_clarify(self, chat_id: int, text: str) -> None:
        """Accept gram clarification, then run the pending food query."""
        try:
            gram = parse_gram(text)
            conv = self._conversation_manager.get(chat_id)
            food_query = conv.pending_food_query or ""
            self._conversation_manager.accept_gram(chat_id, gram)
            await self._analyze_and_send(chat_id, food_query, gram)
        except ValidationError as exc:
            await self._send(chat_id, str(exc))
            await self._send(chat_id, "Ketik jumlah dalam gram, contoh: 150")

    async def _analyze_and_send(self, chat_id: int, food_query: str, gram: float) -> None:
        """Run RAG → LLM → store pending → send with Ya/Tidak buttons."""
        try:
            # ``retrieve_async`` offloads the blocking embed + ChromaDB call
            # to a worker thread, so a stalled Ollama endpoint cannot freeze
            # the bot's event loop (or trip the container healthcheck).
            retrieved = await self._rag_svc.retrieve_async(food_query, 5)
            if not retrieved:
                await self._send(
                    chat_id,
                    f"Tidak ditemukan '{food_query}' dalam basis data TKPI. Coba nama lain.",
                )
                return

            # Get user profile
            user = await self._registration_svc.get_user(chat_id)
            if user is None:
                await self._send(
                    chat_id, "Profil tidak ditemukan. Ketik /start untuk registrasi ulang."
                )
                return

            # Get remaining energy
            energy_record = await self._energy_svc.ensure_today(chat_id, user.tdee_kkal)
            remaining = energy_record.sisa_energi_kkal

            user_profile = UserProfile(
                telegram_id=user.telegram_id,
                jenis_kelamin=user.jenis_kelamin.value,
                usia_tahun=user.usia_tahun,
                berat_kg=user.berat_kg,
                tinggi_cm=user.tinggi_cm,
                tingkat_aktivitas=user.tingkat_aktivitas.value,
                tdee_kkal=user.tdee_kkal,
            )
            result = await self._llm_svc.analyze_food(
                food_name=food_query,
                gram=gram,
                retrieved_foods=retrieved,
                user_profile=user_profile,
                remaining_energy_kkal=remaining,
                user_question=food_query,
            )

            # Compose analysis message
            analysis_text = result.narrative

            # Compute the verified energy/GL payload from TKPI data. The LLM
            # narrative is never the source for callback arithmetic. TKPI
            # nutrition is declared per 100 g of edible portion (BDD); scale the
            # eaten gram by the BDD percentage so the stored figure matches the
            # actual consumed nutrient mass (SPEC §5.3).
            first_food = retrieved[0]
            energi_kkal = None
            if first_food.energi_per_100g is not None:
                bdd = first_food.bdd if first_food.bdd is not None else 100.0
                energi_kkal = nutrition_for_gram(first_food.energi_per_100g, gram, bdd)
            if energi_kkal is None:
                await self._send(
                    chat_id,
                    "Energi pangan tidak tersedia pada sumber TKPI; konsumsi tidak dapat dicatat.",
                )
                return

            gl_aktual = None
            if first_food.gl_value is not None and first_food.serving_size_g is not None:
                gl_aktual = glycemic_load(first_food.gl_value, gram, first_food.serving_size_g)

            pending = await self._energy_svc.create_pending(
                chat_id,
                nama_pangan=food_query,
                gram=gram,
                energi_kkal=energi_kkal,
                gl_aktual=gl_aktual,
            )

            # Send with callback data bound to the persisted pending payload.
            await self._telegram_svc.send_message_with_inline_buttons(
                chat_id=chat_id,
                text=analysis_text,
                inline_keyboard=build_confirm_keyboard(pending.id),
            )
        except Exception:
            log.error("Analysis failed for chat %s", chat_id)
            await self._send(
                chat_id,
                "Maaf, analisis gagal sementara. Silakan coba lagi beberapa saat.",
            )

    # ------------------------------------------------------------------
    # Callback handlers
    # ------------------------------------------------------------------

    async def _handle_confirm_ya(
        self, chat_id: int, pending_id: UUID, message_id: int | None = None
    ) -> None:
        """Log a confirmed consumption, reducing the balance exactly once."""
        try:
            status, record, warned = await self._energy_svc.confirm_consumption(chat_id, pending_id)
        except SQLAlchemyError:
            log.error("Ya confirm failed for chat %s", chat_id)
            await self._telegram_svc.send_message(chat_id, "Maaf, gagal menyimpan. Coba lagi.")
            return

        if status is ConfirmationStatus.CONFIRMED:
            assert record is not None
            if warned:
                await self._telegram_svc.send_message(
                    chat_id,
                    f"✅ Tercatat. Sisa energi hari ini: {record.sisa_energi_kkal} kkal.\n"
                    "Perhatian: batas energi harian tercapai (sisa negatif). "
                    "Konsumsi berlebih tidak disarankan.",
                )
            else:
                await self._telegram_svc.send_message(
                    chat_id,
                    f"✅ Tercatat. Sisa energi hari ini: {record.sisa_energi_kkal} kkal.",
                )
        elif status is ConfirmationStatus.ALREADY_RESOLVED:
            await self._telegram_svc.send_message(chat_id, "✅ Analisis ini sudah tercatat.")
        else:  # STALE or NOT_FOUND
            await self._telegram_svc.send_message(
                chat_id, "Analisis sudah tidak berlaku. Kirim makanan baru untuk analisis."
            )

        if message_id is not None:
            await self._telegram_svc.edit_message_reply_markup(
                chat_id, message_id, inline_keyboard=[]
            )

    async def _handle_confirm_tidak(
        self, chat_id: int, pending_id: UUID, message_id: int | None = None
    ) -> None:
        """Decline a consumption: no balance change, no log row."""
        try:
            status = await self._energy_svc.decline_consumption(chat_id, pending_id)
        except SQLAlchemyError:
            log.error("Tidak confirm failed for chat %s", chat_id)
            await self._telegram_svc.send_message(chat_id, "Maaf, gagal memproses. Coba lagi.")
            return

        if status is ConfirmationStatus.STALE:
            await self._telegram_svc.send_message(
                chat_id, "Analisis sudah tidak berlaku. Kirim makanan baru untuk analisis."
            )
        else:
            await self._telegram_svc.send_message(chat_id, "Baik, tidak dicatat.")

        if message_id is not None:
            await self._telegram_svc.edit_message_reply_markup(
                chat_id, message_id, inline_keyboard=[]
            )

    # ------------------------------------------------------------------
    # IDLE helper
    # ------------------------------------------------------------------

    async def _send_welcome_or_prompt_registration(self, chat_id: int) -> None:
        await self._send(
            chat_id,
            "Selamat datang di DiabetaBot! 🍎\n\n"
            "Saya analisis nutrisi dan kelayakan glikemik makanan berdasarkan profil Anda.\n"
            "Ketik /start untuk daftar dulu.",
        )

    # ------------------------------------------------------------------
    # Registration prompts
    # ------------------------------------------------------------------

    async def _send_registration_gender_prompt(self, chat_id: int) -> None:
        await self._send(
            chat_id,
            "Selamat datang! Saya bantu analisis kebutuhan energi harian Anda.\n\n"
            "Apa jenis kelamin Anda?\n"
            "1️⃣ Laki-laki\n"
            "2️⃣ Perempuan",
        )

    async def _send_registration_age_prompt(self, chat_id: int) -> None:
        await self._send(chat_id, "Berapa usia Anda (dalam tahun)? Masukkan angka, contoh: 25")

    async def _send_registration_weight_prompt(self, chat_id: int) -> None:
        await self._send(chat_id, "Berapa berat badan Anda (kg)? Contoh: 70")

    async def _send_registration_height_prompt(self, chat_id: int) -> None:
        await self._send(chat_id, "Berapa tinggi badan Anda (cm)? Contoh: 170")

    async def _send_registration_activity_prompt(self, chat_id: int) -> None:
        await self._send(
            chat_id,
            "Seberapa aktif aktivitas fisik Anda sehari-hari?\n"
            "1️⃣ Sedentari (jarang/sedikit gerak)\n"
            "2️⃣ Ringan (olahraga ringan 1-3x/minggu)\n"
            "3️⃣ Sedang (olahraga sedang 3-5x/minggu)\n"
            "4️⃣ Berat (olahraga berat 6-7x/minggu)\n"
            "5️⃣ Sangat Berat (pekerjaan fisik berat)",
        )

    async def _send_registration_confirmed(self, chat_id: int) -> None:
        conv = self._conversation_manager.get(chat_id)
        p = conv.profile
        assert p.activity is not None
        await self._send(
            chat_id,
            "✅ Registrasi selesai!\n\n"
            f"Profil Anda:\n"
            f"• Jenis kelamin: {p.gender.value}\n"
            f"• Usia: {p.age} tahun\n"
            f"• Berat: {p.weight_kg} kg\n"
            f"• Tinggi: {p.height_cm} cm\n"
            f"• Aktivitas: {p.activity.value}\n\n"
            "Ketik nama makanan dengan jumlah gram untuk analisis, "
            "contoh: nasi goreng 150g\n"
            "Ketik /start untuk daftar ulang kapan saja.",
        )

    # ------------------------------------------------------------------
    # Low-level send helper
    # ------------------------------------------------------------------

    async def _send(self, chat_id: int, text: str) -> None:
        """Send a plain text message."""
        await self._telegram_svc.send_message(chat_id=chat_id, text=text)
