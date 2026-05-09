"""Charge a Stripe customer a USD amount against their default source.

Usage:
    python charge_customer.py <customer_id> <amount_usd>

Reads STRIPE_API_KEY from the environment. Prints the charge id on
success.
"""

import os
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import stripe


def usd_to_cents(amount_str):
    # parse from string with Decimal to avoid binary-float weirdness
    # like 49.00 -> 4899 or 0.1 + 0.2 != 0.3 nonsense. Stripe wants
    # an integer number of cents.
    try:
        dollars = Decimal(amount_str)
    except InvalidOperation:
        raise ValueError(f"not a valid dollar amount: {amount_str!r}")

    if dollars <= 0:
        raise ValueError("amount must be positive")

    cents = (dollars * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(cents)


def main(argv):
    if len(argv) != 3:
        print("usage: charge_customer.py <customer_id> <amount_usd>", file=sys.stderr)
        return 2

    customer_id = argv[1]
    amount_arg = argv[2]

    if not customer_id.startswith("cus_"):
        print(f"customer id looks wrong (expected cus_...): {customer_id}", file=sys.stderr)
        return 2

    api_key = os.environ["STRIPE_API_KEY"]
    stripe.api_key = api_key

    amount_cents = usd_to_cents(amount_arg)

    try:
        charge = stripe.Charge.create(
            customer=customer_id,
            amount=amount_cents,
            currency="usd",
            description=f"Manual back-office charge ({amount_arg} USD)",
        )
    except stripe.error.StripeError as e:
        # surface the Stripe-side message; user-facing message is
        # usually the most useful thing for the operator
        msg = getattr(e, "user_message", None) or str(e)
        print(f"stripe error: {msg}", file=sys.stderr)
        return 1

    print(charge["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
