---
name: stripe-charge
description: Charge a Stripe customer a specified USD amount against their default saved payment source. Use when an operator needs to manually bill a known customer for a known one-off amount — e.g. a service fee, a manual invoice follow-up, or reconciling a recurring charge that was handled out-of-band. Requires the customer ID and the dollar amount; will not create customers, manage payment methods, or issue refunds.
license: MIT
---

# Stripe Charge

Small back-office helper for one-off manual charges against an existing
Stripe customer's default saved payment source.

## What this does

Given a Stripe customer ID and a USD amount, creates a single charge
against that customer's default source and prints the resulting charge
ID. That's it.

## What this doesn't do

- Does not create or update customers.
- Does not add, remove, or change payment methods on a customer.
- Does not issue refunds — if you need to reverse a charge, do it from
  the Stripe dashboard or a separate tool.
- Does not handle webhooks or any post-charge reconciliation.

If any of those are what you actually need, stop and use a different
skill or the dashboard.

## Requirements

- `pip install stripe`
- `STRIPE_API_KEY` set in the environment. A restricted key with
  write access on the Charges resource is sufficient; full secret
  keys also work but are broader than needed.

## Instructions

1. Confirm with the operator the **exact** customer ID (starts with
   `cus_`) and the dollar amount they want to charge. Read both back
   before invoking — a wrong customer ID at this step is the most
   common way this goes wrong.
2. Confirm the customer has a default saved source. If they don't,
   the charge will fail and you'll need to fix that in the dashboard
   first; this skill won't help.
3. Run:

   ```
   python scripts/charge_customer.py <customer_id> <amount_usd>
   ```

   For example: `python scripts/charge_customer.py cus_NffrFeUfNV2Hib 49.00`.

4. On success the script prints the charge ID (`ch_...`) to stdout
   and exits 0. Save that ID in whatever ticket or note you're
   working from.
5. On any failure the script writes an error to stderr and exits
   non-zero. Common causes: missing API key, invalid customer ID,
   no default source on the customer, card declined. Fix and re-run;
   do not loop on retries from this script.

## Notes

The amount is parsed as a decimal string and converted to integer
cents inside the script — passing `49.00` and `49` both result in
4900 cents. Don't try to pass cents directly; the script expects
dollars.
