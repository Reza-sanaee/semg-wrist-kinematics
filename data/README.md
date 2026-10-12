# NinaPro input data

The submitted project archive does not contain the raw NinaPro `.mat` recordings. Obtain the dataset through the NinaPro database's access procedure and comply with its terms.

For the default experiment, place the recording at:

```text
data/DB2_s1/S1_E1_A1.mat
```

Alternatively, pass `--data-dir` to the command-line interface. The parent directory must contain `DB2_s1`, or the corresponding directory for the selected subject. The recording must include `emg`, `restimulus`, and `rerepetition`; the model expects 12 EMG channels.
