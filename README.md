# SLSDR-Based Recommender System

This repository contains the implementation of a recommender system that uses the **SLSDR** feature selection method combined with collaborative filtering to enhance recommendation accuracy and efficiency.

## Overview
- **Main Script**: The main Python file (`A_Main.py`) orchestrates the execution of the recommender system. It imports and utilizes functions from other Python files in the repository.
- **Configuration Options**: Users can customize the execution by commenting or uncommenting specific code sections in the main script. For example, the computation of metrics such as Precision and Recall is disabled by default but can be enabled by uncommenting the relevant lines in the main file.

## Usage Instructions
1. **Run the Main Script**:
   - Open the `A_Main.py` file.
   - Modify options by commenting/uncommenting code sections as per your requirements (e.g., enable Precision and Recall computation).
   - Execute the script:
     ```bash
     python A_Main.py
     ```

2. **Customize SLSDR Algorithm**:
   - Navigate to the file implementing the SLSDR algorithm.
   - Choose between the two methods for computing `W` by commenting/uncommenting the respective options. The default is the original formula from the paper.

3. **Save Results**:
   - The outputs will be saved in multiple files upon successful execution of the main script. Check the output directory for these files.

4. **Generate Plots**:
   - To visualize the results, run the `plot_results.py` file:
     ```bash
     python plot_results.py
     ```

## Notes
- Ensure all required dependencies are installed before running the scripts.
- Review the code comments for detailed explanations and additional options available.

## Citation
Shang, R., Xu, K., Shang, F. and Jiao, L., 2020. Sparse and low-redundant subspace learning-based dual-graph regularized robust feature selection. Knowledge-Based Systems, 187, p.104830.

