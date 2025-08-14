"""
Test script to verify iterable dataset functionality
"""
import os
import json
from datasets import IterableDataset
from PIL import Image
from config import *

def create_iterable_dataset_generator(json_path, images_base_path="data/images"):
    """Create a generator for iterable dataset"""
    def generator():
        print(f"Loading dataset from {json_path}")
        
        if not os.path.exists(json_path):
            print(f"❌ JSON file not found: {json_path}")
            return
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        processed = 0
        
        # Handle both dict and list formats
        if isinstance(data, dict):
            items = data.values()
        else:
            items = data
            
        for item in items:
            # Get image path - handle different JSON structures
            image_path = None
            if 'image_path' in item:
                image_path = os.path.join(images_base_path, item['image_path'])
            elif 'ImagePath' in item and item['ImagePath']:
                # Handle list of image paths
                image_paths = item['ImagePath']
                if isinstance(image_paths, list) and image_paths:
                    image_path = image_paths[0]  # Use first image
                else:
                    image_path = image_paths
            
            # Skip if no image path or image doesn't exist
            if not image_path or not os.path.exists(image_path):
                print(f"⚠️  Skipping item - image not found: {image_path}")
                continue
                
            # Create conversation format for SFT
            user_message = item.get('question', '')
            
            # Construct the expected response with reasoning and solution format
            reasoning = item.get('heur_reason', item.get('reason', ''))
            answer_letter = item.get('answer', item.get('correct_answer', ''))
            explanation = item.get('explanation', item.get('correct_answer_explanation', ''))
            
            # Format the assistant response
            assistant_response = f"{REASONING_START}\n{reasoning}\n{REASONING_END}\n\n{SOLUTION_START}\n{answer_letter}: {explanation}\n{SOLUTION_END}"
            
            conversation = [
                {
                    "from": "human",
                    "value": f"<image>\n{user_message}"
                },
                {
                    "from": "gpt", 
                    "value": assistant_response
                }
            ]
            
            yield {
                "image": image_path,
                "conversations": conversation
            }
            
            processed += 1
            if processed % 100 == 0:
                print(f"✅ Processed {processed} samples...")
            
            # Limit for testing
            if processed >= 5:
                print(f"🔍 Test complete - processed {processed} samples")
                break
    
    return generator

def test_iterable_dataset():
    """Test the iterable dataset functionality"""
    print("🧪 Testing Iterable Dataset Functionality")
    print("=" * 50)
    
    # Test with training data
    if os.path.exists(TRAIN_JSON):
        print(f"📁 Testing with: {TRAIN_JSON}")
        
        try:
            # Create iterable dataset
            train_generator = create_iterable_dataset_generator(TRAIN_JSON)
            train_dataset = IterableDataset.from_generator(train_generator)
            
            print("✅ Iterable dataset created successfully")
            
            # Test iteration
            print("\n🔄 Testing dataset iteration...")
            sample_count = 0
            
            for sample in train_dataset:
                sample_count += 1
                print(f"\n📋 Sample {sample_count}:")
                print(f"  Image path: {sample['image']}")
                print(f"  Conversations: {len(sample['conversations'])} messages")
                
                # Test image loading
                try:
                    img = Image.open(sample['image']).convert('RGB')
                    print(f"  Image size: {img.size}")
                    print(f"  ✅ Image loaded successfully")
                except Exception as e:
                    print(f"  ❌ Error loading image: {e}")
                
                # Show conversation preview
                for i, msg in enumerate(sample['conversations']):
                    content_preview = msg['value'][:100] + "..." if len(msg['value']) > 100 else msg['value']
                    print(f"  Message {i+1} ({msg['from']}): {content_preview}")
                
                if sample_count >= 3:  # Test first 3 samples
                    break
            
            print(f"\n✅ Successfully tested {sample_count} samples")
            print("🎉 Iterable dataset is working correctly!")
            
        except Exception as e:
            print(f"❌ Error testing dataset: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"❌ Training JSON not found: {TRAIN_JSON}")
        print("Please run download_data.py first")

if __name__ == "__main__":
    test_iterable_dataset()