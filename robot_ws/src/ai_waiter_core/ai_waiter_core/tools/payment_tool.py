from langchain_core.tools import tool
from .payment.payment_mgr import PaymentManager

# Initialize the Payment Manager
payment_provider = PaymentManager()

@tool
def request_payment(table_id: str, amount: float) -> str:
    """
    Generates a payment QR code link for the customer.
    
    table_id: The ID of the table (e.g., 'T1', 'T5')
    amount: The total amount to be paid in VND
    """
    try:
        if amount <= 0:
            return "ERROR: Payment amount must be greater than 0."
            
        qr_link = payment_provider.generate_qr_payload(table_id, amount)
        
        return (
            f"Please scan the QR code to pay {int(amount):,} VND.\n"
            f"Payment Link: {qr_link}"
        )
    except Exception as e:
        return f"ERROR: Failed to generate payment link: {str(e)}"
