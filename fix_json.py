import json

# ✅ تحميل `class_labels.json`
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# ✅ عكس القاموس بحيث تكون الأرقام هي المفاتيح
class_labels = {str(v): k for k, v in class_labels.items()}  

# ✅ حفظ `class_labels.json` بالصيغة الصحيحة
with open("class_labels.json", "w") as f:
    json.dump(class_labels, f, indent=4)

print("✅ تم تصحيح class_labels.json!")