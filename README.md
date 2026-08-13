```powershell
mkdir atr-xai
cd atr-xai

mkdir configs, data, outputs, src, static, templates
mkdir outputs\models, outputs\cam, outputs\report
mkdir static\uploads, static\results

py -3.12 -m venv venv
venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pandas numpy grad-cam flask pyyaml scikit-learn matplotlib

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python src\convert_to_yolo.py

Get-ChildItem data\yolo\images\train | Measure-Object
Get-ChildItem data\yolo\images\validation | Measure-Object
Get-Content configs\data.yaml

python src\train.py

# If training crashes, resume:
Get-ChildItem runs\detect | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python src\train.py --resume runs\detect\train-N\weights\last.pt

Copy-Item runs\detect\train-N\weights\best.pt outputs\models\best.pt

python src\explain.py --weights outputs\models\best.pt --source data\yolo\images\validation --out outputs\cam

python src\generate_report_figures.py --train-results runs\detect\train-N\results.csv

python src\failure_analysis.py --top-n 6 --min-support 5

python app.py
