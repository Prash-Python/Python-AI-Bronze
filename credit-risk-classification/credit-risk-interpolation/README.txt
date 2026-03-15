scikit-learn
pandas


Interpretation & Strategic Analysis1. Identifying High-Risk CustomersBased on the data,
high-risk customers typically share a "Default Profile":Lower Credit Scores: 
Generally below 680.Higher Loan Terms: 10+ years (longer exposure to risk).Employment Type: 
Self-employed individuals in this specific dataset show a higher default rate
(100% in this small sample).2. Patterns Leading to DefaultA clear pattern emerges
 where high Loan-to-Income ratios coupled with declining Credit Scores trigger a default prediction. 
 When a customer's requested loan is nearly equal to or exceeds their annual income, 
 the "distance" to the default cluster in our KNN model shrinks significantly.3. 
 Influence of Credit Score and IncomeCredit Score: Acts as a "proxy" for past behavior. 
 In KNN, it creates a vertical boundary; if you drop below a certain coordinate, 
 your "neighbors" are almost all defaulters.Income: Acts as the "capacity" to pay. 
 High income can sometimes offset a mediocre credit score, but only if the loan amount is modest.4. 
 Suggested Banking PoliciesTiered Interest Rates: Instead of a hard "Yes/No," 
 use the KNN probability. If the model says 66% chance of default, 
 approve the loan but at a higher interest rate (Risk-Based Pricing).Mandatory Collateral: 
 For any applicant classified as "Self-Employed" with a credit score < 660, 
 require a guarantor or collateral.5. KNN vs. 
 Decision TreesFeatureKNNDecision TreeLogicDistance/SimilarityRules (If-Then)ScalingRequiredNot 
 RequiredOutliersSensitiveRobustInterpretability"You are like Person X""You failed Rule Y"6. 
 What if LoanAmount dominates distance?If we don't scale the data, CreditScore (300-900) 
 and LoanAmount (up to 16) will be treated unequally. Since 900 is much larger than 16, 
 the model will essentially ignore the Loan Amount and Term, 
 making decisions almost entirely based on Credit Score. This leads to poor risk assessment.7. 
 Should KNN be used in Real-Time systems?Yes, but with caveats.Pros: 
 It handles new data points instantly without "re-training."Cons: 
 It is computationally expensive (slow) if you have millions of customers, 
 as it must calculate the distance to every record for every new application. 
 For a mid-sized bank, it is excellent for real-time use.
