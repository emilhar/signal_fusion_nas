from datetime import datetime

a = str(datetime.now().replace(microsecond=0)).replace(" ", "_")
print(f"{a}")