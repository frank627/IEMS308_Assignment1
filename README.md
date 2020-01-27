# IEMS Assignment 1
Finding abnormal cluster (outlier) to identify rooms for improvements in Medicare's service and fairness

## How to run 
* Download [Medicare_Provider_Util_Payment_PUF_CY2017.txt](https://drive.google.com/file/d/1FlrzgC0vUllsfoEICzo0uMmHzFnBGlxG/view?usp=sharing) and [converted.csv](https://drive.google.com/file/d/1TgXTiDHi7rL6RSiGCl-GDNa3YSHb7yG1/view?usp=sharing). Put those file in data folder. I could not push those files because the file size of those files exceeded the github limit. 
* Open source_code.ipynb file and run each code snippet. I recommend this way than running source_code.py
* Or run source_code.py.

### Prerequisites
Python (numpy, pandas, csv, copy, sklearn, scipy, matplotlib, seaborn)
Tableau

### What file to check for what
On Canvas announcement, it says that we need to submit 4 things. 
* The source code: check "source_code.ipynb" or "source_code.py". I recommend source code.ipynb because it is easier to read.
* Sample output of the code (as text file): Check "data/clustered_data.csv" or "data/clustered_data.txt". I recommend CSV file because it is more structured
* A half a page 'executive summary' of the findings: Check "Executive Summary.pdf"
* Document with all findings: Check "Report.pdf" 

## File Structure
```
IEMS308_Assignment1
├── README.md 							: This document.
├── source_code.ipynb 						: Code for this assignment. Recommend openning this rather than py file.
├── source_code.py 						: Same as source code.ipynb but in Python format.
├── cluster_analysis.twb 					: Tableau file used to do analysis on clusters.
├── cluster_analysis.pptx 					: Same as twb file but without interactivity.
├── Report.pdf 							: Report on this assignment 1. This also includes Executive Summary.
├── Executive Summary.pdf 					: Copy of Executive Summary of the report.
├── data
│	├── Medicare_Provider_Util_Payment_PUF_CY2017.txt 	: Original data file downloaded from the website. Download it as explained in "How to run" section.
│	├── converted.csv					: Original data file that is converted to CSV. Download it as explained in "How to run" section.
│	├── filtered.csv 					: Data of individual service providers in Wisconsin + aggregated based on npi.
│	├── clustered_data.csv 					: filtered.csv + the column that represents the cluster that each data belongs to. 
│	└── clustered_data.txt 					: Same as clustered_data.csv but in txt format. 
└── img
	├── box_and_whisker.png 				: Box and whisker plot of numerical columns.
	├── heatmap_pearson.png					: Pearson correlation between numerical columns.
	├── distribution_before_transformation.png		: Distribution of numerical columns before transformation.
	├── distribution_after_transformation.png 		: Distribution of numerical columns after transformation.
	├── scree_plot.png					: Scree plot with respect to K from 2 to 20.
	└── silhouette_score.png 				: Silhouette score with respect to K from 2 to 20.
```

## Authors
JunHwa Lee

## Reference
* [http://worldpopulationreview.com/zips/wisconsin/](http://worldpopulationreview.com/zips/wisconsin/)
* [https://www.healthline.com/health/medicare/does-medicare-cover-chiropractic/](https://www.healthline.com/health/medicare/does-medicare-cover-chiropractic)
