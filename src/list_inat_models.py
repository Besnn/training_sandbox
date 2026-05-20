import timm

all_inat = timm.list_models('*inat*', pretrained=True)
print(f"Found {len(all_inat)} models: {all_inat}")

inat21_models = timm.list_models('*_inat21*', pretrained=True)
print(f"iNat21 specific: {inat21_models}")