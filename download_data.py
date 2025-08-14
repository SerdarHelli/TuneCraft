from huggingface_hub import snapshot_download, hf_hub_download
from datasets import Dataset

REXVQA_REPO = "rajpurkarlab/ReXVQA"
REXGRAD_REPO = "rajpurkarlab/ReXGradient-160K"


meta_path = snapshot_download(repo_id=REXGRAD_REPO, repo_type="dataset")

!cat {meta_path}/deid_png.part* > deid_png.tar
!tar -xf ./deid_png.tar
meta_path = snapshot_download(repo_id=REXVQA_REPO, repo_type="dataset")
!cp  {meta_path}/metadata/test_vqa_data.json  /home/QA_json/
!cp  {meta_path}/metadata/train_vqa_data.json  /home/QA_json/
!cp  {meta_path}/metadata/valid_vqa_data.json  /home/QA_json/
