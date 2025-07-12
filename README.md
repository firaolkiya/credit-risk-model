## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord, developed by the Basel Committee on Banking Supervision, is a global regulatory framework that mandates banks to maintain sufficient capital reserves proportional to their risk exposure. It emphasizes **robust risk measurement** and management practices, especially for credit risk, which accounts for a significant portion of a bank’s risk profile.

This regulatory environment requires credit risk models to be **transparent, interpretable, and well-documented** to ensure the models’ reliability and fairness. Interpretability allows risk managers and regulators to understand how input features impact risk predictions, enabling validation, monitoring, and governance of the model. Well-documented models facilitate audits and compliance checks, reducing operational risk and building confidence that the model aligns with regulatory expectations.

Hence, Basel II drives the need for models that not only achieve accurate risk predictions but also provide clear reasoning for their outputs, enabling responsible lending decisions and regulatory approval.

---

### Why is creating a proxy variable necessary given the lack of a direct "default" label, and what are the potential business risks of making predictions based on this proxy?

In traditional credit scoring, a **default label** indicates whether a borrower has failed to meet debt obligations within a specified period. However, in our case, direct labels for default may be unavailable due to limited historical loan performance data or because the eCommerce platform is new to credit lending.

To overcome this, we create a **proxy variable** that approximates default risk by leveraging customer transactional behaviors — such as patterns in Recency (how recently a customer transacted), Frequency (how often they transact), and Monetary value (transaction amounts). These behavioral signals serve as indirect indicators of financial reliability and repayment likelihood.

While this approach enables model training, it introduces **business risks**:

- The proxy may **misrepresent true default risk**, causing inaccurate classification.
- False positives (labeling low-risk customers as high-risk) can lead to lost revenue and customer dissatisfaction.
- False negatives (labeling high-risk customers as low-risk) increase the risk of loan defaults and financial losses.
- The proxy’s quality depends on the assumption that transactional behavior strongly correlates with creditworthiness, which might not always hold.

Therefore, predictions based on proxies require careful validation and continuous monitoring to mitigate risks and ensure decision accuracy.

---

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Credit risk modeling in regulated environments involves balancing **predictive accuracy** with **interpretability and compliance**:

- **Simple, interpretable models**, such as Logistic Regression combined with Weight of Evidence (WoE) encoding, offer several advantages:
  - Their decision-making process is transparent and understandable.
  - They facilitate easy communication of risk drivers to business stakeholders and regulators.
  - These models comply well with Basel II and similar regulations demanding explainability.
  - They allow for straightforward monitoring, validation, and governance.
  - However, they may struggle to capture complex, nonlinear relationships, potentially limiting predictive power.

- **Complex, high-performance models** like Gradient Boosting Machines (GBM), Random Forests, or Neural Networks generally provide:
  - Superior accuracy due to their ability to model intricate patterns and interactions.
  - Better handling of large, high-dimensional datasets.
  - However, their "black box" nature poses challenges for:
    - Regulatory transparency and auditability.
    - Explaining individual credit decisions to customers.
    - Ongoing model validation and governance.
  - Additional tools (e.g., SHAP values, LIME) are required to interpret these models but increase implementation complexity.

In a regulated financial context, the trade-off is a **delicate balance**: while higher predictive power is desirable to reduce defaults and increase profitability, regulators prioritize models that are **understandable, auditable, and fair**. Often, institutions prefer simpler models or hybrid approaches that combine interpretability with acceptable performance, ensuring compliance without sacrificing risk management quality.

---

