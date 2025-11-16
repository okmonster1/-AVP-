import iFeatureOmegaCLI
prot = iFeatureOmegaCLI.iProtein("Datasets/non-AVP/train.txt")

# 正确名字（无下划线，官方暴露名）
for name in ["DistancePair", "CKSAAGP", "QSOrder"]:
    ret = prot.get_descriptor(name)
    if ret is None or not hasattr(prot, "encodings"):
        print(f"{name} 失败")
    else:
        print(f"{name} 成功 -> shape:", prot.encodings.shape)