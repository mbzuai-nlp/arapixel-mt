import os
from pathlib import Path

from datasets import Dataset, DatasetDict

def create_dummy_dataset():
    # Simple Arabic -> English examples
    data = {
        "source": [
            "مرحبا", 
            "كيف حالك؟", 
            "الطقس جميل اليوم", 
            "أنا أحب البرمجة",
            "صباح الخير",
            "هذا اختبار بسيط",
            "قطة نامت على السرير",
            "تعلم الآلة ممتع"
        ],
        "target": [
            "Hello",
            "How are you?",
            "The weather is beautiful today",
            "I love programming",
            "Good morning",
            "This is a simple test",
            "A cat slept on the bed",
            "Machine learning is fun"
        ]
    }
    
    ds = Dataset.from_dict(data)
    
    # Create splits
    dd = DatasetDict({
        "train": ds.select(range(4)),
        "validation": ds.select(range(4, 6)),
        "MSA1": ds.select(range(6, 8))
    })
    
    save_path = Path(os.environ.get("OUTPUT_ROOT", "outputs")) / "dummy_dataset_raw"
    dd.save_to_disk(save_path)
    print(f"Dummy dataset created at {save_path}")

if __name__ == "__main__":
    create_dummy_dataset()
