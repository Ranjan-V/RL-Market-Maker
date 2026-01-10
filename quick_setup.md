# RL Market Maker - Quick Start Guide

## 🚀 Daily Workflow

### Starting Work
```powershell
cd D:\Projects\RL-Market-Maker
.\venv\Scripts\activate
```

### Deactivating (when done)
```powershell
deactivate
```

### Running Scripts
```powershell
# Always activate venv first!
.\venv\Scripts\activate

# Then run your code
python python/training/train_ppo.py
python python/live_trading/paper_trader.py
```

---

## 📦 Package Management

### Install new package
```powershell
pip install package-name
pip freeze > requirements.txt  # Save changes
```

### Check installed packages
```powershell
pip list
```

### Verify specific package
```powershell
python -c "import gymnasium; print(gymnasium.__version__)"
```

---

## 🛠️ Troubleshooting

### Venv not activating?
```powershell
# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again:
.\venv\Scripts\activate
```

### ImportError after restart?
```powershell
# Make sure venv is activated (look for (venv) in prompt)
# If not there:
.\venv\Scripts\activate
```

### Fresh install needed?
```powershell
# Delete venv and reinstall
rm -r venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Project Milestones

- [ ] Week 1-2: Core environment working
- [ ] Week 3-4: RL agents trained
- [ ] Week 5-6: C++ integration
- [ ] Week 7: Live trading on testnet
- [ ] Week 8: Documentation & polish

---

## 💡 Pro Tips

1. **Always check venv is active** - Look for `(venv)` in prompt
2. **Git commit frequently** - Track your progress
3. **Test on small data first** - Don't train on full dataset initially
4. **Keep notes** - Document hyperparameters that work
5. **Monitor GPU usage** - `nvidia-smi` for CUDA utilization

---

## 🔗 Useful Commands

```powershell
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# List all files in project
tree /F

# Check disk space
Get-PSDrive

# Find large files
Get-ChildItem -Recurse | Sort-Object Length -Descending | Select-Object -First 10
```