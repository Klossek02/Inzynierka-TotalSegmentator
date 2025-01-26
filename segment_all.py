# === segment_all.py ===

if __name__ == "__main__":
    from pathlib import Path
    from totalsegmentator.python_api import totalsegmentator


    def predict_all_cts(input_dir, output_dir, start_patient="s0000"):
        """
        Predicts segmentations for all CT files starting from a specific patient ID.

        Args:
            input_dir (str): Path to the input directory containing subject folders.
            output_dir (str): Path to the directory where output segmentations will be saved.
            start_patient (str): The patient ID to start processing from.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # starting patient number from the string
        start_patient_num = int(start_patient[1:])

        subjects = [d for d in input_dir.iterdir() if d.is_dir()]

        # filtering subjects starting from the given patient
        subjects = [
            subject for subject in subjects
            if subject.name.startswith("s") and int(subject.name[1:]) >= start_patient_num
        ]

        # sorting subjects by their order
        subjects.sort(key=lambda x: int(x.name[1:]))

        for subject in subjects:
            input_ct = subject / "ct.nii.gz"
            if not input_ct.exists():
                print(f"CT file not found for subject: {subject}")
                continue

            output_seg = output_dir / subject.name
            output_seg.mkdir(parents=True, exist_ok=True)

            print(f"Segmenting: {input_ct}")
            totalsegmentator(input_ct, output_seg, fast=True)

        print(f"Predictions saved to: {output_dir}")

    predict_all_cts(
        input_dir=r"/Totalsegmentator_dataset_v201", 
        output_dir="/Predictions", 
        start_patient="s0000"
    )
