
import plotly.graph_objects as go


def getPlotHistPValues(bins_centers, p_values_hist, title, errors=None):
    fig = go.Figure()
    if errors is not None:
        trace = go.Bar(x=bins_centers, y=p_values_hist,
                       error_y=dict(type="data", array=errors))
    else:
        trace = go.Bar(x=bins_centers, y=p_values_hist)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="p value", yaxis_title="count", title=title)
    return fig
