# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

## Install and run

Use Python 3.12 on Windows and create a virtual environment from the repository
root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The lock file records the exact dependency versions tested for this environment.
In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter. To
activate the environment in a PowerShell terminal, run
`.\.venv\Scripts\Activate.ps1`.

The pipeline also needs the MIALab dataset; it is not included in this
repository. By default, it uses `..\data\atlas`, `..\data\train`, and
`..\data\test`. If those directories are unavailable, it also checks
`..\MIAlab_data\MIALab_2020_Dataset`. For a dataset stored elsewhere, provide
its `atlas`, `train`, and `test` directories explicitly:

```powershell
.\.venv\Scripts\python.exe pipeline.py `
  --data_atlas_dir "C:\path\to\dataset\atlas" `
  --data_train_dir "C:\path\to\dataset\train" `
  --data_test_dir "C:\path\to\dataset\test"
```

The output is written to `mia-result` by default. Change it with
`--result_dir`. After intentionally changing dependencies in `requirements.txt`,
install them in `.venv` and regenerate `requirements-lock.txt` with
`.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements-lock.txt`.

The pipeline compares the simplified CRF baseline (`compat=2.0`) with
per-label removal of 26-connected components smaller than 25 mm³. It reports
paired per-tissue Dice and HD95 results; an arm passes the comparison gate when
at least one tissue's mean HD95 improves and no tissue's mean Dice drops by
more than 0.005. Results include `*_SEG-PP.mha` and
`*_SEG-PP-components-25mm3.mha`, plus `postprocessing_per_subject.png` with
per-subject Dice and HD95 plots.

The separate DenseCRF comparison arm runs only when `pydensecrf` is installed;
otherwise the pipeline reports it as unavailable instead of substituting the
simplified CRF. To limit peak memory, test subjects are processed one at a time
and forest predictions are computed in chunks.

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.
