import scipy.io
import pandas as pd

# Load the data from the .mat file
mat_data = scipy.io.loadmat('data/array/psy/array.mat')

# Access the MyData structure in the .mat file
# Adjust 'MyData' depending on how it's named in your .mat file


MyData = mat_data['MyData']

df = pd.DataFrame()

# Number of trials
NumTrials = 600

for i in range(NumTrials):
    trial = MyData[i]
    arraycate = trial['arraycate'][0]  # Access arraycate from the trial data
    targetcate = trial['targetcate'][0][0]  # Access targetcate from the trial data

    # Find the ground truth position of the target category
    gtpos = [idx + 1 for idx, cate in enumerate(arraycate) if cate == targetcate]

    # Print the trial number and ground truth position
    print(f"Trial number: {i + 1}; Ground truth position is: {gtpos[0] if gtpos else 'Not Found'}")
    df = df._append({'Trial': i + 1, 'Ground Truth Position': gtpos[0] if gtpos else 'Not Found'}, ignore_index=True)

df.to_csv('data/array/gt_positions.csv', index=False)

