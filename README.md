# HumberAI-ML

This Lab checks whether a loan is affordable based on a 40% debt-to-income limit, for anyone estimating loan eligibility from their income and existing debt.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate

No external packages are required; the script only uses the Python standard library.

## Run

    python3 "Lab 01.py"

## Example

    Enter loan amount ($1,000 - $1,000,000): 20000
    Enter down payment ($0 - $20,000.00): 2000
    Enter net monthly income ($500 - $100,000): 5000
    Enter monthly debt ($0 - $5,000.00): 1200

    Amount to finance: $18,000.00
    Remaining monthly debt capacity: $800.00
    Debt-to-income ratio: 24.0%
    Result: Approved: debt-to-income ratio is well within the 40% limit

## Known limitations

Only two results are returned (Approved or Denied). Range limits (e.g. max loan of $1,000,000) are estimates and not tied to any real lending policy.