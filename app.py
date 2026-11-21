"""shiny run app.py   ->  Daisy demo (UK motor insurance dataset).

To use your own data:
    from daisy.app import make_app
    app = make_app(my_dataframe, dataset_name="mydata", starting_response="claims_cost",
                   kpi_spec="mydata_kpi_spec.csv", feature_spec="mydata_feature_spec.csv",
                   filter_spec="mydata_filter_spec.csv")
"""
from daisy.app import demo_app

app = demo_app()
