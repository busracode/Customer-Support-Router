"""Application rules separate from frozen training preprocessing."""
import re
from src.preprocessing import normalize_text

GROUPS = {
    "Orders & Shopping": "order order_status shopping_list shopping_list_update",
    "Banking & Cards": "account_blocked apr balance card_declined credit_limit credit_limit_change credit_score damaged_card direct_deposit exchange_rate expiration_date freeze_account improve_credit_score interest_rate international_fees new_card order_checks pin_change redeem_rewards replacement_card_duration report_fraud report_lost_card rewards_balance routing spending_history transactions transfer",
    "Billing & Insurance": "bill_balance bill_due insurance insurance_change min_payment pay_bill",
    "Travel & Transport": "book_flight book_hotel car_rental carry_on current_location directions distance flight_status international_visa lost_luggage plug_type share_location traffic travel_alert travel_notification travel_suggestion uber",
    "Reservations": "accept_reservations cancel_reservation confirm_reservation how_busy restaurant_reservation restaurant_reviews restaurant_suggestion",
    "Vehicle Service": "gas gas_type jump_start last_maintenance mpg oil_change_how oil_change_when schedule_maintenance tire_change tire_pressure",
    "HR & Payroll": "application_status income payday pto_balance pto_request pto_request_status pto_used rollover_401k taxes w2",
    "Food & Recipes": "calories cook_time food_last ingredient_substitution ingredients_list meal_suggestion nutrition_info recipe",
    "Device & Assistant Support": "alarm calendar calendar_update change_accent change_ai_name change_language change_speed change_user_name change_volume find_phone make_call meeting_schedule next_song play_music reminder reminder_update repeat reset_settings schedule_meeting smart_home sync_device text timer todo_list todo_list_update update_playlist user_name what_song whisper_mode",
    "General Assistance": "are_you_a_bot calculator cancel date definition do_you_have_pets flip_coin fun_fact goodbye greeting how_old_are_you maybe meaning_of_life measurement_conversion next_holiday no roll_dice spelling tell_joke thank_you time timezone translate vaccines weather what_are_your_hobbies what_can_i_ask_you what_is_your_name where_are_you_from who_do_you_work_for who_made_you yes",
}
TEAMS = {label: team for team, labels in GROUPS.items() for label in labels.split()}

def route(label, confidence, threshold):
    if label == "oos":
        return "Human Agent", "Out-of-scope prediction"
    if confidence is None or confidence < threshold:
        return "Human Agent", "Below confidence threshold"
    return TEAMS.get(label, "Human Agent"), "Accepted prediction" if label in TEAMS else "Unmapped intent"

def small_talk(text):
    text = normalize_text(text).strip(" .!?")
    for pattern, reply in [
        (r"(?:hi|hello|hey|good morning|good afternoon|good evening)", "Hello! How can I help you today?"),
        (r"(?:thanks|thank you|thanks a lot)", "You're welcome!"),
        (r"(?:bye|goodbye|see you)", "Goodbye! Have a nice day."),
        (r"(?:what is your name|who are you|are you a bot)", "I'm a support routing assistant. Tell me what you need help with."),
        (r"i feel (?:sad|upset|frustrated|angry|unhappy)", "I'm sorry you feel that way. Can you tell me what happened?"),
    ]:
        if re.fullmatch(pattern, text):
            return reply
    return None

def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        nxt = [i]
        for j, right in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = nxt
    return row[-1]

def prepare_message(text, correct_products=False):
    codes = re.findall(r"\b[A-Za-z]{1,5}-?\d{3,8}(?:-[A-Za-z0-9]+)?\b", text)
    corrections = []
    products = ("dishwasher", "refrigerator", "freezer", "microwave", "washing", "vacuum")
    def replace(match):
        word = match.group()
        if not correct_products or len(word) < 6 or word.lower() in products:
            return word
        ranked = sorted((edit_distance(word.lower(), product), product) for product in products)
        if ranked[0][0] <= (2 if len(word) >= 8 else 1) and ranked[0][0] < ranked[1][0]:
            corrections.append((word, ranked[0][1]))
            return ranked[0][1]
        return word
    corrected = re.sub(r"\b[a-zA-Z]+\b", replace, text)
    return normalize_text(corrected), codes, corrections
