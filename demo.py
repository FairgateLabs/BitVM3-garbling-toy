import garblings.bitmv3_garbling as bitmv3_garbling
import garblings.goat_garbling as goat_garbling
import garblings.bitmv3_garbling_linear as bitmv3_garbling_linear


print("Checking BitVM3 Garbling -----------------------------------------")
if bitmv3_garbling.attack():
    print("BitVM3 Garbling attack succeeded.")
else:
    print("BitVM3 Garbling attack failed.")

print()
print("Checking Goat Garbling -----------------------------------------")
if goat_garbling.attack():
    print("Goat Garbling attack succeeded.")
else:
    print("Goat Garbling attack failed.")



print()
print("Checking BitVM3 Linear Adaptors Garbling -----------------------------------------")
if bitmv3_garbling_linear.attack():
    print("BitVM3 Garbling attack succeeded.")
else:
    print("BitVM3 Garbling attack failed.")

