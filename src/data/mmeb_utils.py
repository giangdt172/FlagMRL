"""Target instructions shared by the MMEB train and evaluation splits."""

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    **dict.fromkeys([
        "ImageNet_1K", "ImageNet-1K", "HatefulMemes", "SUN397", "N24News", "VOC2007",
        "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    ], POS_MOD_CLASS_LABEL),
    **dict.fromkeys([
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W",
        "ScienceQA", "GQA", "TextVQA", "VizWiz",
    ], POS_MOD_ANSWER),
    **dict.fromkeys(["MSCOCO_i2t", "VisualNews_i2t"], POS_MOD_IMAGE_CAPTION),
}
