SELECT
    CASE
        WHEN tenure < 12 THEN '0-12 months'
        WHEN tenure < 24 THEN '12-24 months'
        WHEN tenure < 48 THEN '24-48 months'
        ELSE '48+ months'
    END AS tenure_segment,
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY tenure_segment, Contract
ORDER BY churn_rate DESC;
