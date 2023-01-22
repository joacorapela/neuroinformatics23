
import plotly.graph_objects as go


def getPlotHistPValues(bin_centers, p_values, title, errors=None):
    fig = go.Figure()
    if errors is not None:
        trace = go.Bar(x=bin_centers, y=p_values,
                       error_y=dict(type="data", array=errors))
    else:
        trace = go.Bar(x=bin_centers, y=p_values)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="p value", yaxis_title="count", title=title)
    return fig
