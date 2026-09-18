def get_valid_number(prompt, min_value, max_value):
   
    while True:
        raw_value = input(prompt)
        try:
            value = float(raw_value)
        except ValueError:
            print(f"Invalid input: '{raw_value}' is not a number. Please enter a number.")
            continue

        if value < min_value or value > max_value:
            print(f"Number is out of range: value must be between {min_value} and {max_value}.")
            continue

        return value


def loan_affordability_check(loan_amount, down_payment, net_monthly_income, monthly_debt):
    max_debt_to_income = 0.40
    max_allowable_debt = net_monthly_income * max_debt_to_income
    remaining_debt_capacity = max_allowable_debt - monthly_debt
    financed_amount = loan_amount - down_payment
    debt_to_income_ratio = monthly_debt / net_monthly_income

    if debt_to_income_ratio > max_debt_to_income:
        result = "Denied: existing debt already exceeds the 40% debt-to-income limit"
    else:
        result = "Approved: debt-to-income ratio is well within the 40% limit"

    print("")

    print(f"Amount to finance: ${financed_amount:,.2f}")
    print(f"Remaining monthly debt capacity: ${remaining_debt_capacity:,.2f}")
    print(f"Debt-to-income ratio: {debt_to_income_ratio:.1%}")
    print(f"Result: {result}")
    


def main():
    loan_amount = get_valid_number("Enter loan amount ($1,000 - $1,000,000): ", 1000, 1000000)
    down_payment = get_valid_number(f"Enter down payment ($0 - ${loan_amount:,.2f}): ", 0, loan_amount)
    net_monthly_income = get_valid_number("Enter net monthly income ($500 - $100,000): ", 500, 100000)
    monthly_debt = get_valid_number(f"Enter monthly debt ($0 - ${net_monthly_income:,.2f}): ", 0, net_monthly_income)

    loan_affordability_check(loan_amount, down_payment, net_monthly_income, monthly_debt)


if __name__ == "__main__":
    main()

