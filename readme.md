# Parity8：最重要的實驗指令

適用於 AMD Instinct MI300X／ROCm 6.4.3 的 Linux 容器。
以下命令都在專案根目錄執行；AMD GPU 在 PyTorch 中也使用 `--device cuda`。

## 1. 第一次準備環境

如果 Bash 尚未初始化 Conda：

```bash
conda init bash
source ~/.bashrc
```

建立 Python 環境並安裝套件（已有 parity8 環境就跳過 create）：

```bash
conda create -n parity8 python=3.12 pip -y
conda activate parity8
python -m pip install -r requirements.txt
```

之後每次登入只需 `conda activate parity8`，不必重裝。
同步專案到遠端時，請另外複製整個 `data/`；Git 不會傳送被忽略的 data/、run/、doc/。

## 2. 檢查 GPU 與模型

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm:', torch.version.hip); print('GPU available:', torch.cuda.is_available())"
python src/parity8_align_20260924.py --selfcheck --device cuda
```

確認 GPU available 為 True，且 selfcheck 正常完成。這兩條不會啟動完整訓練。

## 3. 資料準備（已有快取則驗證）

```bash
for task in parity linearly_separable mnist_pca mnist_pca_small hidden_manifold two_curves
do
    python src/experiment_tasks.py --task "$task" || break
done
```

這一步不訓練。沒有資料時會自動生成或下載；有有效快取就直接讀取。
若有錯誤，先修正再開始實驗。

## 4. 一次依序跑六個任務

先預覽清單，不訓練：

```bash
python src/run_experiments.py --batch full_run --device cuda --dry-run
```

正式開始：

```bash
python -u src/run_experiments.py --batch full_run --device cuda --checkpoint-every 100
```

- 任務與順序由根目錄 `experiments.json` 控制。
- 每個任務預設 6 種模型 × 9 個 p 設定 × 40 個 seed。
- 每 100 個完整 epoch 定期保存，另在首次成功、預算邊界與完成時保存。
- **同名 batch 再次執行會續訓，已完成的訓練設定不重跑。**
- 第一次使用批次腳本會建立新 run，先前手動啟動的 run 不會自動併入。

想全部重新訓練，換一個批次名稱：

```bash
python -u src/run_experiments.py --batch repeat_02 --device cuda
```

## 5. 用 tmux 執行，SSH 斷線仍繼續

```bash
tmux new -s parity8
```

在 tmux 裡啟用環境，確認位於專案根目錄後執行：

```bash
conda activate parity8
python -u src/run_experiments.py --batch full_run --device cuda --checkpoint-every 100
```

- 離開畫面但保留執行：先按 **Ctrl+B**，放開後按 **D**。
- 回到画面：`tmux attach -t parity8`。
- **Ctrl+C 會要求安全保存並停止整批實驗。**
- 停止後重跑相同 batch 命令，即可接續剩餘實驗。

不要同時啟動兩個相同 batch。主機／容器重啟後需重新執行命令續訓。

## 6. 單獨跑一個任務與續訓

例如只跑 Two curves：

```bash
python -u src/parity8_align_20260924.py --task two_curves --tag single_run --device cuda --checkpoint-every 100
```

可用 task：`parity`、`linearly_separable`、`mnist_pca`、`mnist_pca_small`、
`hidden_manifold`、`two_curves`。

**這種單獨啟動命令每次都建立新 run。** 要續訓請直接複製程式印出的 --resume 命令，
或將下方 REPLACE_WITH_RUN_ID 換成實際資料夾名稱：

```bash
resume_run='run/two_curves/single_run/REPLACE_WITH_RUN_ID'
python -u src/parity8_align_20260924.py --resume "$resume_run" --device cuda
```

## 7. 結果在哪裡

批次實驗：`run/<task>/full_run__<name>/<run_id>/`。
單獨實驗：`run/<task>/<tag>/<run_id>/`。

| 位置 | 內容 |
| --- | --- |
| results/ | 主結果 CSV、逐 seed CSV、各組 JSON |
| weights/ | 首次成功的模型權重 |
| checkpoints/ | 中斷續訓的 latest／previous 狀態 |
| progress.json | 各組進度與保存 epoch |

批次總進度：

```bash
cat run/batches/full_run/batch.json
```

完整說明：[環境與資料準備](doc/SETUP_MI300X.md)、[實驗指令（含 nohup）](doc/EXPERIMENT_COMMANDS.md)、
[新增批次任務](doc/BATCH_EXPERIMENTS.md)。
