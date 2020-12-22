# ROSSMANN Sales Forecasting: An End-to-End Data Science Project Using Machine Learning for Predictions

<img src="https://www.rossmann.de/dam/jcr:f83a6bfd-e46c-4063-844e-59ffa529ed80/Buehne_Startseite_Ueber_uns.2017-01-11-08-36-49.jpg" alt="drawing" width="100%"/>

You are provided with historical sales data for 1,115 Rossmann stores.

This repository contains files from the project I'm building <a href="https://www.hotmart.com/product/data-science-em-producao/Y24207977W">for this course</a>.

Data source: <a href="https://www.kaggle.com/c/rossmann-store-sales/data">Rossmann Store Sales | Kaggle</a>

---
## Contents <p id="contents"></p>
- <a href="#intro">A Brief Introduction to the ROSSMANN Company</a>
- <a href="#crisp">Project Methodology: CRISP-DM</a>
- <a href="#p1">Phase 1: Business Understanding</a>
  - <a href="#buss">The Business Context</a>
- <a href="#p2">Phase 2: Data Understanding</a>
    - <a href="#s1">Step 1: Data Collection and Description</a>
    - <a href="#s2">Step 2: Hypothesis Creation and Feature Engineering</a>
    - <a href="#s3">Step 3: Filtering Variables and Rows</a>
    - <a href="#s4">Step 4 Exploratory Data Analysis (EDA)</a>
- <a href="#p3">Phase 3: Data Preparation</a>
    - <a href="#s5">Step 5: Data Preparation</a>
    - <a href="#s6">Step 6: Feature Selection</a>
- <a href="#p4">Phase 4: Modeling</a> 
    - <a href="#s7">Step 7: Machine Learning Modeling</a>
    - <a href="#s8">Step 8: Hyperparameter Fine Tuning</a>
- <a href="#p5">Phase 5: Evaluation</a> 
    - <a href="#s9">Step 9: Translating and Interpreting the Error</a>
- <a href="#p6">Phase 6: Deployment</a> 
    - <a href="#s10">10. Deploying Machine Learning Model to Production: a telegram bot</a>
- <a href="#conclusion">Conclusion</a>

--- 
## A Brief Introduction to the ROSSMANN Company <p id="intro"></p>

This is based on the ROSSMANN company portrait available on [the company website](https://www.rossmann.de/unternehmen/ueber-uns/unternehmensportrait.html)

Dirk Rossmann GmbH is one of the largest drug store chains in Europe and the largest in Germany in 2020 ([see list](https://www.statista.com/statistics/505614/number-of-drugstore-branches-germany/)). The retail company, founded by Dirk Roßmann in 1972 in Germany, operates over 3,000 drug stores in 7 European countries. The company is increasing the number of stores outside of Germany over the last years, as shown by [this chart](https://www.statista.com/statistics/1009654/number-of-rossmann-stores-by-country/).

With more than 4,000 branches (2,196 of them in Germany), the company operations extend to Albania, Czech Republic, Hungary, Poland, and  Turkey. The foreign companies contributed 30 percent to group sales.

The group started the year of 2020 intending to open 200 new branches and intended to maintain the rate of expansion (a volume of investiment of 200 million euros).

ROSSMANN has around around 21,700 different items with a focus on skin and body care, food and luxury foods, baby, detergents, cleaning and hair care.

<a href="#contents">Back to Contents</a>

---
## Project Methodology: CRISP-DM <p id="crisp"></p>

The method used manage the project was the Cross-Industry Standard Process for Data Mining (CRISP-DM). This is one of the most used techniques for Data Science Projects. 

As said by [Wirth and Hipp](http://www.cs.unibo.it/~danilo.montesi/CBD/Beatriz/10.1.1.198.5133.pdf), "the CRISP-DM reference model for data mining provides an overview of the **life cycle** of a data mining project". This process allows us to iterate over the steps and we can map all possible problem in the project.

Aiming to provide more productivity and effectiveness, the Data Science project is broken in six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

### CRISP-DM Cycle
<img src="img/crispdm.PNG" alt="drawing" width="100%"/>

The main reason why CRISP-DM was used is because it has four main advantages:

* for each complete cycle, we have and end-to-end solution that can be implemented and attend some of business demands;
* it provides a more quickly business results than many others methodologies;
* we can map many problems and antecipate impediments and avoid them;
* it helps us not to spend to much time on a particular step.

In this project, the CRISP-DM phases is broken into 10 steps: Business Understanding, Data Description, Feature Engineering, Filtering Variables, Exploratory Data Analysis, Data Preparation, Feature Selection, Machine Learning Modeling, HyperParameters Fine Tuning, Translating and Interpreting the Error, and the Model Deployment.

Merging the phases and the steps we have:

#### Phase 1: Business Understanding
Correctly identify and understand the business demand objectives and the requirements from a business perspective. This understanding has to be deeper, identifying the true stakeholder and reason of the request.

* **Step 0: Business Demand**

#### Phase 2: Data Understanding
It starts with the data collection and then with activities in order to get familiar with the data, to identify data quality problems, to discover some initial insights into the data, and/or to detect interesting subsets to form hypotheses for hidden information.

* **Step 1: Data Collection and Description**
* **Step 2: Hypothesis Creation and Feature Engineering**
* **Step 3: Filtering Variables and Rows**
* **Step 4: Exploratory Data Analysis**

#### Phase 3: Data Preparation
This phase aims to prepare the data for modeling.

* **Step 5: Data Preparation (transforming, scaling, ...)**
* **Step 6: Feature Selection**

#### Phase 4: Modeling
Build Machine Learning models and compare them based on Cross validation technique. After that, choose the best one and tune its parameters.

* **Step 7: Machine Learning Modeling**
* **Step 8: Hyperparameter Fine Tuning**

#### Phase 5: Evaluation
Evaluate the model results with some appropriate metrics. Besides that, translate that metrics to the business field.

* **Step 9: Translating and Interpreting the Error**

#### Phase 6: Deployment
Create the model and evaluate it is not usually the end of the project. The results have to be deliver or presented to the stakeholders. So, that's what this phase is about.

* **Step 10: Deploying the Machine Learning Model to Production: a telegram bot**

#### Next Cycle
After completing all these phases, we have a solution (even though it cannot be the best) to satisfy the initial business demand. After that, we evaluate the project outcome, the business needings, and then decide if there's need to continue with more cycle. If so, all the steps is done again in order to improve the model.

[back to contents](#contents)

---
## Phase 1: Business Understanding <p id="p1"></p>

Before starting the data analysis and modeling, the first task is to understand the business problem we received as a data scientist. That's important because there's a big chance that the request we received is not exactly what the stakeholder wants (_and sometimes the person who made the request for us is no the true stakehold. In that case, we also have to identify who really is the stakeholder_).

For that goal, we need to seek to understand four things:
1. The context behind the business request: how did it come about?
2. The reason the person is making the request;
3. Who is the main stakeholder of the request. If is someone other than the managers, maybe the request is not exactly that we received; Besides that, the main stakeholder can guide us in the project;
4. The solution format:
      - granularity: daily, weekly, by stores, by product, and so forth;
      - What kind of machine learning problem it is: classification, regression, clustering, and so on;
      - Main methods that could be used: time series, SMV, RNA, and so on;
      - presentation method: dashboard, mail, smartphone message, and so on.

### The Business Context <p id="buss"></p>

⚠️⚠️ _Disclaime:_ Since we don't work for ROSSMANN and we have only the dataset, it'll be helpful to create a hypothetical business context to guide the project. The following context is basically a mix between the <a href="https://www.kaggle.com/c/rossmann-store-sales/data">Rossmann description and request of the dataset on Kaggle</a> and some additions from the <a href="https://www.hotmart.com/product/data-science-em-producao/Y24207977W">Data Science em Produção</a> course.
So, let's pretend we work as data scientists for ROSSMANN.

Our Data Science team received the request from the store managers to forecast the sales for its respective stores for the next six weeks. Before starting to handle the data, our team decided to better understand that business request. We found that in fact the request came up from the CFO who asked the managers for the predictions of the stores revenue for the next six weeks because he wanted to reform the stores and he wanted to aticipate the store amount of sales after six weeks as the investment.

So, our team started to understand that four things that was said before:
- **The business request context:** In a monthly meeting, the CFO requested the managers the next six weeks sales predictions for each store;
- **The reason behind teh request:** the CFO wants to anticipate a portion of the stores revenue to invest in the stores reformation;
- **Stakeholder:** CFO
- **The Solution format:**
    - **Granularity:** daily sales by stores;
    - **Kind of problem:** Sales Forecast;
    - **Main methods:** Regression, Time Series;
    - **Delivering Method:** Real-time six weeks sales forecasting in a smartphone app.

[back to contents](#contents)

---
## Phase 2: Data Understanding <p id="p2"></p>

After better understand the business demand, our team could start handling with data. As said before, this phase involves loading, cleaning, applying descriptive statistics, and exploring the data. This phase comprises steps one through four.

### Step 1: Data Collection and Description <p id="s1"></p>
In this step we can see how challenging is the problem we are dealing with. 

#### Data Collection
In a real-life project, this step starts with requests on databases, APIs, and so on in order to collect all the information available that could help us solve the problem. With that in hand, our team can create a final dataset. Since this project works with data from Kaggle, our first task in this step is just to load the csv dataset.

The data is from <a href="https://www.kaggle.com/c/rossmann-store-sales/data">Rossmann Store Sales on Kaggle</a>. There's three main datasets:
- Training Data: historical data for training the model (it includes the target: sales)
- Test Data: historical data for testing the model (it includes the target: sales)
- Store Data: supplemental information about the stores

#### Training Data Dimensions:

    - Number of Rows: 1,017,209
    - Number of Columns: 18
    - Date Range: from 2013-01-01 to 2015-07-31

#### Variables Description and types:
table from [this repository](https://github.com/alanmaehara/Sales-Prediction)

| Variable      | Description | Data Type |
| ----------- | ----------- | ----------- |
| **sales** (target)  | **the turnover for any given day (this is what you are predicting)** | **numerical (continuous)**   |
| store   | Store ID (unique) | numerical (discrete)  |
| day_of_week   | day of the week (1 = Monday, 7 = Sunday) | numerical (discrete)    |
| date  | date of each sales entry | date  |
| customers  | the number of customers on a given day  | numerical (discrete) |
| open   | an indicator for whether the store was open: 0 = closed, 1 = open | numerical (dummy)  | 
| promo  | indicates whether a store is running a promo on that day | numerical (dummy)   |
| state_holiday   | indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None | categorical (nominal)  |
| school_holiday  | indicates if the (Store, Date) was affected by the closure of public schools: 1 = affected, 0 = not affected | numerical (dummy*)  |
| store_type   | differentiates between 4 different store models: a, b, c, d | categorical (nominal)  |
| assortment | describes an assortment level: a = basic, b = extra, c = extended | categorical (ordinal)  |
| competition_distance   | distance in meters to the nearest competitor store | numerical (continuous)   |
| competition_open_since_month  | gives the approximate month of the time the nearest competitor was opened | numerical (discrete)  |
| competition_open_since_year   | gives the approximate year of the time the nearest competitor was opened | numerical (discrete) | 
| promo2  | promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating | numerical (dummy)  |
| promo2_since_week   |describes the calendar week when the store started participating in Promo2 | numerical (discrete) |
| promo2_since_year   |describes the year when the store started participating in Promo2 | numerical (discrete) |
| promo_interval   |describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store | categorical (nominal) |

#### Data Cleaning: Imputation and Changing Types
Along the this step, our team identified some missing values and some inappropriate data types. It's important to deal with them because many Machine Learning algoriths can't deal with such problem.  
First, we changed teh 'date' columsn type from object to pandas datetime using the `.astype()` function.

Second, we dealt with the missing data. We identified the following columns with missing values:

| Variable      | NANs |
| ----------- | ----------- | 
| competition_distance | 2642
| competition_open_since_month | 323348
| competition_open_since_year | 323348
| promo2_since_week | 508031
| promo2_since_year | 508031
| promo_interval | 508031

The first task after identifing those NANs would be to try to discover why there are missing values in the data set. In a real-life context, we would have talked to the data engineering team to better identify the reason of the existence of the missing data. Also, it could be a mistake when collecting the data. Since in this fictional project there's no one who could give us information, we cannot know exactly the reason for those NANs. Thus, the next step is to handle it.

There are some techniques to deal with missing values, like input the column mean, drop the rows or the columns, and the use of Machine Learning algoriths to predict the value. However, we tried to input variables based on the business understanding, using other columns the derive a value. Since we are working in CRISP-DM cycles, we could better handle it in the next one. Due to the high number of missing values, the traditional methods could not be so good because if we drop of input the mean we could lose information.

The inputation was as follow:
- **competition_distance:** maybe null values indicates that thre're no competitors next. So, we inputed a value grater thatn the maximum distance in the column. Since the maximum is 75,860m, we inputed 200,000m to the missing data;
- **competition_open_since_month** and **competition_open_since_year**: for both, it was inputed the year from the column `date`;
- **promo2_since_week** and **promo2_since_year**: if there are missing values in these columns, maybe it's because the store is not participating in promo2. So, we'll fill it out with the present date.
- **promo_interval:** I'm going to create the column `is_promo` (dummy variable) to indicate whether a store is holding consecutive promo sales on that day and the `promo_interval` will be dropped in Step 3.

#### Descriptive Statistics
To summarize the data, we used the descriptive statistics. We devided the dataset between numeric and categorical variables and performed the respective statistics techniques.

For **numerical variables**, we basically used the central tendency (mean and median) and the dipersion (Standard Deviation, Minimum, Maximum, Range, First and Third Quartiles, Skewness, and Kurtosis) measures. 

![](img/numericDesc.PNG)

Two columns are highly skewd: `competition_distance` and `competition_open_since_year`. Besides that, `competition_distance` is too large, which can indicates the presence of outliers.

Since there's no much technique to analyze categorical data, we used boxplots for them and sales.

![](img/categDesc.PNG)

It seems that sales tends to differ between store types and the assortment level. 

### Step 2: Hypothesis Creation and Feature Engineering <p id="s2"></p>
To better understand the relationship between the target variable (sales) and the features, we created some hypothesis based on the business problem to guide the feature engineering and then the exploratory data analysis.

The hypothesis created in this step will be validated in the Exploratory Data Analysis step. It let us have a good notion about the relation between the features and the target and it will help us to choose more accurately what feature to use when modeling.

#### Hypothesis Mind Map
to guide the hypotehsis creation, the following mind map was created. The Hypothesis Mind Map has basically three elements:
- The phenomenon: it's what we want to measure or model (sales);
- Agents: entities that somehow impact the phenomenon 9customers, stores);
- Agent's Attributes: age, marritage, size, and so on.

<img src="img/HypMindMap.png" alt="drawing" width="100%"/>

-> Mind Map goal: derive a list of hypotheses and then we can prioritize some of them

#### Hypothesis Creation <p id="hyp"></p>
After creating a hypothesis list, we prioritize some over others. The criterion used was the availability of the corresponding feature.

**Hypothesis Final List**
1. Stores with a larger assortment should sell more.
2. Stores with closer competitors should sell less.
3. Stores with longer competitors should sell more

4. Stores with active promotions for longer should sell more.
5. Stores with more promotion days should sell more.
6. Stores with more consecutive promotions should sell more.

7. Stores open during the Christmas holiday should sell more.
8. Stores should be selling more over the years.
9. Stores should sell more in the second half of the year.
10. Stores should sell more after the 10th of each month.
11. Stores should sell less on weekends.
12. Stores should sell less during school holidays.

#### Feature Engineering
First of all, why do we need to do it and why to do it before the EDA? 
- To have the variables available to study in the exploratory data analysis;
- To don’t have a mess in the EDA section with feature creation, maps, tables, more maps. That is, to have a clean code with feature creation and exploratory analysis in different sections.

We created the following features:
- from `date` column we created `year`, `month`, `day`, `week_of_year`, and `year_week` since we need them to validate some hypothesis;
- `competition_since`: it stands for how long the competition exists considering the purchase date in terms of months. ;
- `promo_since`: how long is there an active promotion.

Some categorical featured had their attribute classes renamed: `state_holiday`, `assortment` was just letters and then became descriptions (like 'a' in state holiday that became 'public_holiday').

Since `store_type` had no description in the data source, it wasn't changed
.
### Step 3: Filtering Variables and Rows <p id="s3"></p>
Based on business restrictions (e.g. we won't have a variable avaliable in the future to use in the prediciton), we have to properly filter variables.

This step is divided into two partes: Filtering Rows and Filtering Columns.

**Filtering Rows:** since it's obvious that closed stores has no sales in that day, we dropped rows where `open` is equals to zero. Also, we considered only the rows where `sales` is greater than zero.

**Filtering Columns:**
- `customers`: we can't use customers because we won't have this data available for the next six weeks (the prediction period) unless we build another project to predict how many customers the stores will have in that period;
- `promo_interval` and `month_map` were used to create new columns. They'll not be used anymore.
- since `open` has no use anymore, we dropped it.

### Step 4: Exploratory Data Analysis (EDA)<p id="s4"></p>
In this step we go deeper in the data to get some valuable business insights. Basically, we want to know which variables is the most important to the prediction and what is the strength of that importance.

The EDA is divided into three parts
1. Univariate Analysis: to get a overview of the individually variables, looking at their distribution and counting classes for categorical features;
2. Bivariate Analysis: check the relationship between the features and the target based on the <a href="#hyp">Hypothesis list</a>. This is one of the most important steps, which we can get some business insights and decide if the feature is really important to the model.
3. Multivariate Analysis: check the relationship between the features and identify those highly correlated.

#### Univariate Analysis
It's divided into Response, Numerical, and Categorical analysis.

- **Response Variable (sales):** the target distribution seems to be non Normal. Since the majority of Machine Learning algorithms needs the target to be normal, we checked that by using both Q-Q plot and Shapiro-Wilk test for normality. As shown bt the Q-Q plot bellow, the distributions don't seems to be follow a normal distribution. Shapiro-Wilk test (statistics = 0.902; p-value = 0.000) confirms that it is probably non Gaussian. Therefore, we'll have to rescale it before performing the model.

![](img/response.PNG)
![](img/responseqq.PNG)

- **Numerical Variables:** all variables seem not to be normal. Some Highlights: 
        - `day_of_week`: sales are lower on Sundays;
        - `competition_distance`: the are more stores with competitors close than far;
        - `promo2_since_year`: there are more stores that joined consecutive promotion sales in 2013.
        
![](img/edanumeric.PNG)

- **Categorical Variables:** 
    - `state_holiday`: There's a difference for the three types of holiday. Based on that difference, the model could better adjust its predictions. For instance, it's christmas period;
    - `store_type`: for different type of stores, sales have a different behavior. For instance, store of type b have sales less concentrated than the others;
    - `assortment`: there are fewer sales for stores with 'extra' assortment than with the other classes.

![](img/edacateg.PNG)

#### Bivariate Analysis
- **H1: ** 
![](img/h1.PNG)
![](img/hh1.PNG)

- **H2: ** 
![](img/h2.PNG)

- **H3: ** 
![](img/h3.PNG)

- **H4: ** 
![](img/h4.PNG)

- **<s>H5:</s>** 
- **H6: ** 
![](img/h6.PNG)

- **H7: ** 
![](img/h7.PNG)

- **H8: ** 
![](img/h8.PNG)

- **H9: ** 
![](img/h9.PNG)

- **H10: ** 
![](img/h10.PNG)

- **H11: ** 
![](img/h11.PNG)

- **H12: ** 
![](img/h12.PNG)



#### Multivariate Analysis


![](img/multicateg.PNG)





[back to contents](#contents)