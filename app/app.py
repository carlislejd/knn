import pandas as pd
from flask import Flask, render_template, request


from .model import input_punk, cum_graph



df = pd.read_csv('app/punks.csv', index_col=0)

def create_app():
    app = Flask(__name__)

        
    @app.route('/', methods=["GET", "POST"]) 
    def ml():
        """ 
        Our about us page.
        """
        if request.method == "GET":
            assetId = list(df.index.astype(str))
            return render_template('input.html', data=assetId)
        
        if request.method == "POST":
            input = request.form.get("input")
            model, final, usd_mean, eth_mean = input_punk(df, int(input))
            link = 'https://www.larvalabs.com/cryptopunks/cryptopunk' + input + '.png'
            cum = cum_graph(final)
            return render_template('output.html', output_song=input, tables=[final.to_html(classes='data', header="true", index=False)], 
                                   punk_model=model, original=link, cum=cum, avg_usd=usd_mean, avg_eth=eth_mean)
    
    return app
    