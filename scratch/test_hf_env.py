import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import huggingface_hub

print("HF_HUB_DISABLE_SSL_VERIFICATION:", os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION"))
# Let's inspect some of the internal constants of huggingface_hub to see what config is loaded.
try:
    from huggingface_hub.constants import HF_HUB_DISABLE_SSL_VERIFICATION as const_val
    print("HF_HUB_DISABLE_SSL_VERIFICATION constant:", const_val)
except Exception as e:
    print("Failed to import constant:", e)

# Also let's print all variables in constants that have SSL in their name
import huggingface_hub.constants as hf_const
for attr in dir(hf_const):
    if "SSL" in attr or "VERIFY" in attr or "CA" in attr:
        print(f"Constant {attr}: {getattr(hf_const, attr)}")
