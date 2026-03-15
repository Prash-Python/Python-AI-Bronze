# Employee Productivity Predictor
A Multivariate Linear Regression model to predict employee performance based on experience, training, and workload.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the model: `python productivity_model.py`

## Business Logic
The Model Framework: Multivariate Linear RegressionTo predict the Productivity Score (
Y
), we use a linear combination of the independent variables (
X
1
,
X
2
,
.
.
.
). The relationship is defined by the following equation:$$Y = \beta_0 + \beta_1(\text{Exp}) + \beta_2(\text{Train}) + \beta_3(\text{WorkHrs}) + \beta_4(\text{Proj}) + \epsilon$$Where:$\beta_0$ is the intercept (base productivity).$\beta_1, \beta_2, \beta_3, \beta_4$ are the coefficients (weights) for each feature.$\epsilon$ represents the error term.2. Data Interpretation & AnalysisLooking at the trends in your dataset (e.g., as Training Hours go from 20 to 90, the Score jumps from 55 to 92), we can derive the following insights:Which factor most strongly impacts productivity?Statistically, we look at the Standardized Coefficients. In this specific dataset, Training Hours and Experience show the strongest positive correlation with the score. However, Training Hours often shows the most immediate "boost" per unit in these types of workforce models.How does training affect productivity?Training has a positive linear relationship with productivity. Based on the sample, for roughly every 20-hour increase in training, the productivity score increases by approximately 10–12 points. It acts as a skill multiplier.Should the company increase training hours or working hours?Increase Training Hours. While both show increases, the data suggests that Working Hours have a diminishing return and physical limits. Training improves the quality of output per hour, which is more sustainable for workforce planning than simply increasing the quantity of hours.What happens if Working Hours increase beyond optimal limits?In a real-world scenario (and advanced non-linear models), we see the "Law of Diminishing Returns." * Burnout: Productivity per hour drops.Error Rates: Fatigue leads to more mistakes, requiring more time for "rework," which eventually drags the score down despite the high hours.Can productivity ever decrease with more experience?In this linear model, no—it’s a straight line up. However, in reality, it can decrease due to Skill Obsolescence (if the employee doesn't keep up with new tech) or Disengagement. To model this, we would use a polynomial regression (
X
2
) to show a curve that might plateau or dip.3. Model Health & ImprovementHow to detect Overfitting?Overfitting happens when the model "memorizes" the noise in your 10 rows of data rather than learning the actual trend.Train/Test Split: Split your data. If the model is 99% accurate on the 10 rows but 60% accurate on new data, it's overfit.$R^2$ vs. Adjusted 
R
2
: If you add useless features and 
R
2
 goes up but Adjusted 
R
2
 stays the same, you are overfitting.Suggested New Feature: "Peer Feedback Score" or "Tool Proficiency"I’d suggest adding "Well-being/Engagement Score." * Why: High experience and high hours mean nothing if the employee is disengaged. Adding a qualitative metric helps capture the "human" element that raw hours cannot.
