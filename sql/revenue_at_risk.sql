SELECT
    Contract,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS churned_monthly_revenue,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) / SUM(MonthlyCharges),
        2
    ) AS revenue_churn_rate
FROM customers
GROUP BY Contract
ORDER BY revenue_churn_rate DESC;
