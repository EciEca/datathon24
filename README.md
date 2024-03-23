## Devpost
https://devpost.com/software/vineyard-ventures

## Inspiration
Driven by a shared passion for making data exploration more interactive and accessible to all, we were inspired to create a tool that reflected the intricacies and nuances of red wine. The project aims to bridge the gap between traditional wine tasting methods and modern data analysis techniques. By leveraging machine learning and interactive visualization, the dashboard provides a user-friendly platform for both experts and novices to delve deeper into the complexities of red wine characteristics. Ultimately, the project aspires to enhance wine appreciation, empower producers, and enrich the overall wine experience for consumers.

## What it does
Vineyard Ventures, a web-based interactive dashboard, analyzes the quality of red wine based on various attributes like acidity, alcohol content, and more. It utilizes the Dash framework, Plotly library, and scikit-learn for machine learning functionalities. 

The dashboard features components like dropdown menus and sliders to visualize attribute distributions, correlations, and predict wine quality. Upon selecting an attribute, the program dynamically updates the plots and provides detailed descriptions of each attribute's significance in determining wine quality. Additionally, it employs machine learning models, particularly Random Forest Classifier, to predict whether input attribute values correspond to good or bad quality wine. 

## How we built it
Vineyard Ventures embodies a collection of all group members’ unique ideas. Our consensus led us to devise a strategy centered around a dynamic dashboard that is both educational and interactive.

Transitioning to the dashboard's development phase, we meticulously tailored its layout to resonate with our intended audience. Careful consideration was given to color palettes and fonts, ensuring visual appeal aligned with our project's objectives. Every feature was thoughtfully orchestrated. For instance, incorporating sliders enhances user experience, facilitating clear visualization of prediction values. Leveraging Plotly, we integrated a hovering functionality into graphs, permitting users to access essential statistical data such as mean, minimum, and maximum values of each attribute.

For our predictor, we used a random forest classifier that uses a bagging technique to train multiple decision trees independently. The predictions from each tree are then aggregated through a voting mechanism to determine the final prediction. The reason we used a random forest is because of its ability to handle the dataset's complex relationships and to reduce overfitting. Additionally, we tuned the hyper-parameter settings, such as the number of trees, to optimize the algorithm's accuracy.

In bridging the frontend and backend, we employed SingleStore as a pivotal tool. This platform served as our testing ground for graphs, streamlining our workflow significantly. Given our team's limited familiarity with GitHub, SingleStore emerged as an intuitive solution for seamless code sharing, strengthening our collaborative efforts effectively.


## Challenges we ran into
Our journey in developing our project presented numerous hurdles that we faced head-on. Initially, we started on the challenging task of learning Dash and SingleStore from scratch. Simultaneously, we delved into the complexities of machine learning models, requiring meticulous attention to detail. However, one of the most significant challenges we encountered was the need to distinguish our project from others working with the same dataset. To address this, we devised an innovative solution: an interactive dashboard tailored for wine enthusiasts to explore and learn more about their favorite beverages. Despite these obstacles, our dedication to innovation remained unwavering, allowing us to explore alternative strategies to make our project stand out.


## Accomplishments that we're proud of
We are particularly proud of learning how to use Dash and Plotly for our frontend development as well as SingleStore for sharing code. These technologies were unfamiliar territory for us initially, with none of our team members having prior experience in dashboard creation. However, with the help of online tutorials, the SingleStore mentors through discord, and many hours of work we've harnessed these once-daunting tools to craft a polished and impactful project.


## What we learned
Participating in this project enabled our team to acquire a multitude of skills and gain numerous lessons. In regards to the skills, we practiced the art of crafting a dashboard utilizing Dash and SingleStore. We explored various graphing techniques and plot styles before arriving at the creation of an informative and interactive end product. As for lessons, we deepened our comprehension of the debugging process in python. We also realized that it was also crucial for us to invest ample time in brainstorming the optimal approach for dataset analysis because comprehensive comprehension of each attribute is imperative in crafting the most informative dashboard for users. 


## What's next for Vineyard Ventures
1. Facilitating user input for detailed reviews on various wines sourced globally, thereby enriching the dataset utilized by the dashboard.
2. Developing a model with enhanced accuracy compared to the existing one through rigorous training.
3. Providing users with refined wine suggestions post-prediction based on their preferences.
4. Integrating multiple datasets to augment the complexity and richness of attributes accessible to users.
