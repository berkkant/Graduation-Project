# Graduation-Project
In our project, we want to predict people's sleep quality by applying state-of-the-art deep learning techniques and making it federated. We are using NetHealth open-source dataset, which was collected from 698 students in total during eight semesters. This comprehensive study includes data on communication patterns from mobile phones, sleep, and physical activity routines, students’ family backgrounds, living conditions, personality, etc., from surveys. This study uses some parts of the basic survey and all the wearable device data. Therefore, preprocessing is applied to extract the required data to construct a sub-dataset for the analysis.

# Execution
In order to run this FL architecture, we need to first run the server
python3 server.py <port_number>
and then we need to open different terminals for each client and run these commands for each terminal.
python3 client<client_number> <port_number>
Then the FL process will start.

