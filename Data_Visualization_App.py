import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import StringIO, BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Gentex PP1', layout='wide')
st.title('Gentex PP1 Data Web App 📈')
st.subheader('Feed me with the transformed Excel file')

uploaded_file = st.file_uploader('Choose a XLSX file', type='xlsx')

if uploaded_file is not None:
    # Load the Excel file
    excel_file = pd.ExcelFile(uploaded_file)

    # Create a dropdown menu
    option = st.selectbox('Choose an option', ['Mechanical', 'Optical', 'Electrical'])
    # Set sheet_names based on the selected option
    if option == 'Mechanical':
        sheet_names = excel_file.sheet_names[:6]
    elif option == 'Optical':
        sheet_names = excel_file.sheet_names[6:12]
    elif option == 'Electrical':
        sheet_names = excel_file.sheet_names[12:]
    # Get the sheet names
    #sheet_names = excel_file.sheet_names
    
    # Create a dropdown menu
    sheet = st.selectbox('Choose a sheet', sheet_names)
    
    # Read the selected sheet
    data = pd.read_excel(uploaded_file, sheet_name=sheet)

    # If the selected sheet is 'Mechanical_Raw_data' and 'Date' column exists, convert it to datetime and extract the month in 'YYYY-MM' format
    #if sheet == sheet_names[0] or sheet == sheet_names[1] and 'Date' in data.columns:
    if ('Raw' in sheet or 'DPTM' in sheet) and 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Create multiselects for 'Part', 'Type', and 'Measurements'
    if 'Part' in data.columns:
        unique_parts = data['Part'].unique().tolist()
        selected_parts = st.multiselect('Select Part', unique_parts, default=unique_parts)
        data = data[data['Part'].isin(selected_parts)]
        
    if 'Type' in data.columns:
        unique_types = data['Type'].unique().tolist()
        selected_types = st.multiselect('Select Type', unique_types, default=unique_types)
        data = data[data['Type'].isin(selected_types)]
        
    if 'Measurements' in data.columns:
        unique_measurements = data['Measurements'].unique().tolist()
        selected_measurements = st.multiselect('Select Measurements', unique_measurements, default=unique_measurements[0])
        data = data[data['Measurements'].isin(selected_measurements)]
    
    # Display the data
    st.dataframe(data.reset_index(drop=True))

# Add a divider
st.markdown("---")

# Add a title and subheader
st.header('Data Viz: Box-Plots (Left vs Right)')
st.subheader('Grouping by (Parts) per month, Select a Measurement:')

# Create a dropdown menu for columns after the 4th column from the first sheet
if uploaded_file is not None:
    # Read the first sheet as strings
    first_sheet_data = pd.read_excel(uploaded_file, sheet_name=sheet_names[0], dtype=str)
    
    # If 'Date' column exists, convert it to datetime and extract the month in 'YYYY-MM' format
    if 'Date' in first_sheet_data.columns:
        first_sheet_data['Date'] = pd.to_datetime(first_sheet_data['Date'], format='%Y%m%d')
        first_sheet_data['Date'] = first_sheet_data['Date'].dt.strftime('%Y-%m')
    
    column_names = first_sheet_data.columns.tolist()[4:]
    selected_column = st.selectbox('Select a column', column_names, key= 'selected_column')
    
    # Melt the columns after the 4th column
    id_vars = first_sheet_data.columns.tolist()[:4]
    melted_data = pd.melt(first_sheet_data, id_vars=id_vars, value_vars=column_names, var_name='Measurements', value_name='Data').sort_values(by='Part', ascending=True)
    
    # Convert the 'Data' column to numeric
    melted_data['Data'] = pd.to_numeric(melted_data['Data'], errors='coerce')
    
    # Display the melted data based on the selected column
    with st.expander('View Raw Data', expanded=False):
        st.dataframe(melted_data[melted_data['Measurements'] == selected_column].sort_values(by=['Date','Part', 'Type'], ascending=[True,True, True]).reset_index(drop=True))

    # Read the 'DPM Mechanical_Stats_Summary' sheet
    stats_summary = pd.read_excel(uploaded_file, sheet_name= sheet_names[4])
    
    # Get the USL and LSL values for the selected column
    usl = stats_summary.loc[stats_summary['Measurements'] == selected_column, 'USL'].values[0]
    lsl = stats_summary.loc[stats_summary['Measurements'] == selected_column, 'LSL'].values[0]

    # Create a box plot
    colors = ['rgb(7,40,89)', 'rgb(9,56,125)', 'rgb(8,81,156)', 'rgb(107,174,214)']
    unique_dates = melted_data['Date'].unique()
    # Get the column names
    col2_column_names = stats_summary.columns.tolist()
    # Create a list of default columns
    col2_default_columns = ['Date', 'Part', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'CpK']
    # Create a multiselect widget for the columns
    col2_selected_columns = st.multiselect('Select from the Statistical Summary Columns', col2_column_names, default=col2_default_columns)
    # Create three columns
    col1, col2, col3 = st.columns([1.8, 1, 0.8])
    # Create sliders and a dropdown menu in the third column
    with col3:
        #st.subheader("Chart Effects")
        st.markdown("#### Chart Effects")
        jitter = st.slider('Jitter: 0.3 ~ default', 0.0, 1.0, 0.3, key='jitter')
        whiskerwidth = st.slider('Whisker Width: 0.2 ~ default', 0.0, 1.0, 0.2, key='whiskerwidth')
        marker_size = st.slider('Marker Size: 3 ~ default', 1, 10, 3, key='marker_size')
        line_width = st.slider('Line Width: 1 ~ default', 1, 10, 1, key='line_width')
        pointpos = st.slider('Point Position: -1.8 ~ default', -2.0, 2.0, -1.8, key='pointpos')
        boxpoints_options = ['outliers','all', 'suspectedoutliers']
        boxpoints = st.selectbox('Box Points: all ~ default', boxpoints_options, index=1, key='boxpoints')
        if st.button('Reset Chart Effects'):
            jitter = 0.3
            whiskerwidth = 0.2
            marker_size = 3
            line_width = 1
            pointpos = -1.8
            boxpoints = 'all'
        # Create a box plot in the first column
    with col1:
        selected_dates = st.multiselect('Select dates:', unique_dates, default=unique_dates, key= 'box1 dates')
        unique_parts_0 = melted_data['Part'].unique()
        selected_parts_0 = st.multiselect('Select Parts:', unique_parts_0, default= unique_parts_0, key= 'box1 parts')
        plt.clf()  # Clear the current figure
        fig = go.Figure()
        # Create a dictionary that maps each unique date to an index
        date_to_index = {date: index for index, date in enumerate(selected_dates)}
        for date in selected_dates:
            fig.add_trace(go.Box(
                x=melted_data[(melted_data['Date'] == date) & (melted_data['Measurements'] == selected_column) & (melted_data['Part'].isin(selected_parts_0))]['Part'],
                y=melted_data[(melted_data['Date'] == date) & (melted_data['Measurements'] == selected_column) & (melted_data['Part'].isin(selected_parts_0))]['Data'],
                name= date,
                boxpoints=boxpoints,
                jitter=jitter,
                whiskerwidth=whiskerwidth,
                marker_size=marker_size,
                line_width=line_width,
                pointpos=pointpos,
                marker_color=colors[date_to_index[date] % len(colors)],
                line_color=colors[date_to_index[date] % len(colors)]
            ))
        # Add horizontal lines for USL and LSL
        all_parts = melted_data['Part'].unique()
        fig.add_trace(go.Scatter(
            x=all_parts, 
            y=[usl]*len(all_parts), 
            mode='lines', 
            name='USL', 
            line=dict(color='red', width=1, dash='dash'),
            visible='legendonly'  # Make the line unselected by default
        ))
        fig.add_trace(go.Scatter(
            x=all_parts, 
            y=[lsl]*len(all_parts), 
            mode='lines', 
            name='LSL', 
            line=dict(color='red', width=1, dash='dash'),
            visible='legendonly'  # Make the line unselected by default
        ))
        # Determine the y-axis range
        min_value = min(melted_data[(melted_data['Date'].isin(unique_dates)) & (melted_data['Measurements'] == selected_column)]['Data'].min(), lsl)
        max_value = max(melted_data[(melted_data['Date'].isin(unique_dates)) & (melted_data['Measurements'] == selected_column)]['Data'].max(), usl)
        fig.update_layout(
            title_text=f"Box Plot: {selected_column}<br><sub>USL: {usl}, LSL: {lsl}</sub>", 
            title_font=dict(size=18), 
            showlegend=True,
            boxmode='group',  # group together boxes of the different traces for each value of x
        )
        fig.update_yaxes(range=[min_value, max_value])
        st.plotly_chart(fig)

    # Display the dataframe in the second column
    with col2:
        # Display the selected columns of the transposed dataframe
        st.markdown("##### Statistical Summary:")
        # Create a multiselect widget for the dates
        selected_dates = st.multiselect('Select dates', unique_dates, default=unique_dates)
        # Filter the dataframe based on the selected dates
        filtered_stats_summary = stats_summary[(stats_summary['Measurements'] == selected_column) & (stats_summary['Date'].isin(selected_dates))]
        col2_df = filtered_stats_summary.reset_index()[col2_selected_columns].transpose()
        #col2_df = stats_summary[stats_summary['Measurements'] == selected_column].reset_index()[col2_selected_columns].transpose()
        with st.expander('View Statistical Summary table', expanded=False):
            st.dataframe(col2_df)
    st.markdown("- - -")
    # Create three columns
    col4, col5, col6 = st.columns([0.6, 0.7, 1])
    with col5:
        st.markdown("##### Density Plot Filters:")
        selected_dates= st.multiselect('Select dates', unique_dates, default=unique_dates, key= 'datescol5')
        unique_parts_1 = melted_data['Part'].unique()
        selected_parts_1= st.multiselect('Select Parts', unique_parts_1, default= unique_parts_1, key='partscol5')
    with col4:
        st.markdown(f"#### Density Plot: {selected_column}")
    #plt.figure(figsize=(4, 3))
        for date in selected_dates:
            for part in selected_parts_1:
                part_data = melted_data[(melted_data['Measurements'] == selected_column) & (melted_data['Part'] == part) & (melted_data['Date'] == date)]['Data']
                sns.set(style="darkgrid")
                sns.distplot(part_data, hist=True, kde=True, bins=int(15), hist_kws={'edgecolor':'blue'},
                             kde_kws={'shade': True, 'linewidth': 3}, label=part +  ' '+ date[5:])
    # Get the y-axis limits
    _, ymax = plt.ylim()

    with col6:
        st.markdown("##### Legend and Text Parameters:")
        bbox_x = st.slider('Legend x position: 1.25 ~ default', 0.0, 2.0, 1.25, key='bbox_x')
        bbox_y = st.slider('Legend y position: 1.02 ~ default', 0.0, 2.0, 1.0, key= 'bbox_y')
        vertical_alignment_options = ['top', 'bottom', 'baseline', 'center_baseline']
        horizontal_alignment_options = ['center', 'left', 'right']
        va = st.selectbox('Text vertical alignment: bottom ~ default', vertical_alignment_options, index=0, key='va')
        ha = st.selectbox('Text horizontal alignment: left ~ default', horizontal_alignment_options, index=0, key='ha')
        text_y = st.slider('Text y position:', 0.0, ymax*2, ymax, key='text_y')
        if st.button('Reset Chart Effects', key='legend Part chart'):
            bbox_x = 1.25
            bbox_y = 1.02
            va = 'bottom'
            ha= 'left'
            text_y= ymax
    with col4:
        # Add vertical lines for USL and LSL
        plt.axvline(x=usl, color='r', linestyle='--', label='USL'+ ' ' + str(usl))
        plt.text(usl, text_y, 'USL: ' + str(usl), color='r', va=va, ha=ha)
        if lsl != 0:
            plt.axvline(x=lsl, color='r', linestyle='--', label='LSL'+ ' '+ str(lsl))
            plt.text(lsl, text_y, 'LSL: ' + str(lsl), color='r', va=va, ha=ha)
        plt.legend(prop={'size': 10}, loc='upper right', bbox_to_anchor=(bbox_x, bbox_y)) #, title = 'Legend')
        plt.title(f'Density Plot')
        plt.xlabel('Data')
        plt.ylabel('Density')
        st.pyplot(plt)
        plt.clf()

# Add a divider
st.markdown("---")

# Add a title and subheader
st.header('Data Viz: Box-Plots (Rel vs Non Rel)')
st.subheader('Grouping by (Part, Type) per month, Select a Measurement:')

# Check if a file has been uploaded
if uploaded_file is not None:
    # Create a dropdown menu for columns after the 4th column from the first sheet
    # The selected value will be stored in the variable 'selected_column_1'
    selected_column_1 = st.selectbox('Select a column', column_names, key= 'selected_column_1')

    # Read the 'DPM Mechanical_Stats_Summary' sheet from the uploaded file
    stats_summary_1 = pd.read_excel(uploaded_file, sheet_name= sheet_names[2])
    # Get the USL and LSL values for the selected column
    usl_1 = stats_summary_1.loc[stats_summary_1['Measurements'] == selected_column_1, 'USL'].values[0]
    lsl_1 = stats_summary_1.loc[stats_summary_1['Measurements'] == selected_column_1, 'LSL'].values[0]
    
    # Create a copy of the melted_data DataFrame
    melted_data_1 = melted_data.copy()

    # Create a new 'Part_Type' column in the copied DataFrame
    # This column is a combination of the 'Part' and 'Type' columns, separated by a space
    melted_data_1['Part_Type'] = melted_data_1['Part'] + ' ' + melted_data_1['Type']

    # Display the rows of the copied DataFrame where the 'Measurements' column matches the selected column
    with st.expander('View Raw Data', expanded=False):
        st.dataframe(melted_data_1[melted_data_1['Measurements']== selected_column_1].sort_values(by=['Date', 'Part','Type'], ascending=[True,True,True]).reset_index(drop=True))
    
    # Get the column names
    col_column_names = stats_summary_1.columns.tolist()
    # Create a list of default columns
    col_default_columns = ['Date', 'Part', 'Type', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'CpK']
    # Create a multiselect widget for the columns
    col_selected_columns = st.multiselect('Select from the Statistical Summary Columns', col_column_names, default=col_default_columns, key='col_selected_columns')
    # Display the selected columns of the transposed dataframe
    st.markdown("##### Statistical Summary:")
    # Filter the dataframe based on the selected dates
    filtered_stats_summary_1 = stats_summary_1[(stats_summary_1['Measurements'] == selected_column_1) & (stats_summary_1['Date'].isin(selected_dates))]
    col_df = filtered_stats_summary_1.reset_index()[col_selected_columns]
    with st.expander('View Statistical Summary table', expanded= False):
        st.dataframe(col_df)
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    # Create sliders and a dropdown menu in the third column
    with col2:
        #st.subheader("Chart Effects")
        st.markdown("#### Chart Effects")
        jitter = st.slider('Jitter: 0.3 ~ default', 0.0, 1.0, 0.3, key='jitter_1')
        whiskerwidth = st.slider('Whisker Width: 0.2 ~ default', 0.0, 1.0, 0.2, key='whiskerwidth_1')
        marker_size = st.slider('Marker Size: 3 ~ default', 1, 10, 3, key='marker_size_1')
        line_width = st.slider('Line Width: 1 ~ default', 1, 10, 1, key='line_width_1')
        pointpos = st.slider('Point Position: -1.8 ~ default', -2.0, 2.0, -1.8, key='pointpos_1')
        boxpoints_options = ['outliers','all', 'suspectedoutliers']
        boxpoints = st.selectbox('Box Points: all ~ default', boxpoints_options, index=1, key='boxpoints_1')
        if st.button('Reset Chart Effects', key='chart_2'):
            jitter = 0.3
            whiskerwidth = 0.2
            marker_size = 3
            line_width = 1
            pointpos = -1.8
            boxpoints = 'all'
    # Create a box plot in the first column
    with col1:
        selected_dates = st.multiselect('Select dates:', unique_dates, default=unique_dates, key= 'box2 dates')
        unique_types_2 = melted_data_1['Part_Type'].unique()
        selected_types_2 = st.multiselect('Select Parts:', unique_types_2, default= unique_types_2, key= 'box2 PartType')
        plt.clf()  # Clear the current figure
        fig = go.Figure()
        date_to_index = {date: index for index, date in enumerate(selected_dates)}
        for date in selected_dates:
            fig.add_trace(go.Box(
                x=melted_data_1[(melted_data_1['Date'] == date) & (melted_data_1['Measurements'] == selected_column_1) & (melted_data_1['Part_Type'].isin(selected_types_2))]['Part_Type'].sort_values(ascending=True),
                y=melted_data_1[(melted_data_1['Date'] == date) & (melted_data_1['Measurements'] == selected_column_1) & (melted_data_1['Part_Type'].isin(selected_types_2))]['Data'],
                name= date,
                boxpoints=boxpoints,
                jitter=jitter,
                whiskerwidth=whiskerwidth,
                marker_size=marker_size,
                line_width=line_width,
                pointpos=pointpos,
                marker_color=colors[date_to_index[date] % len(colors)],
                line_color=colors[date_to_index[date] % len(colors)]
            ))
        # Add horizontal lines for USL and LSL
        all_parts = melted_data_1['Part_Type'].unique()
        fig.add_trace(go.Scatter(
            x=all_parts, 
            y=[usl_1]*len(all_parts), 
            mode='lines', 
            name='USL', 
            line=dict(color='red', width=1, dash='dash'),
            visible='legendonly'  # Make the line unselected by default
        ))
        fig.add_trace(go.Scatter(
            x=all_parts, 
            y=[lsl_1]*len(all_parts), 
            mode='lines', 
            name='LSL', 
            line=dict(color='red', width=1, dash='dash'),
            visible='legendonly'  # Make the line unselected by default
        ))
        # Determine the y-axis range
        min_value = min(melted_data_1[(melted_data_1['Date'].isin(unique_dates)) & (melted_data_1['Measurements'] == selected_column_1)]['Data'].min(), lsl_1)
        max_value = max(melted_data_1[(melted_data_1['Date'].isin(unique_dates)) & (melted_data_1['Measurements'] == selected_column_1)]['Data'].max(), usl_1)
        fig.update_layout(title_text=f"Box Plot: {selected_column_1}<br><sub>USL: {usl_1}, LSL: {lsl_1}</sub>", title_font=dict(size=18), showlegend=True,
                          boxmode='group',  # group together boxes of the different traces for each value of x
                         )
        fig.update_yaxes(range=[min_value, max_value])
        st.plotly_chart(fig)
    st.markdown("- - -")
    # Create three columns
    col7, col8, col9 = st.columns([0.6, 0.7, 1])
    with col8:
        st.markdown("##### Density Plot Filters:")
        unique_dates_1 = melted_data_1['Date'].unique()
        selected_dates_1= st.multiselect('Select dates', unique_dates_1, default=unique_dates_1, key= 'datescol8')
        unique_types_1 = melted_data_1['Part_Type'].unique()
        selected_types_1 = st.multiselect('Part_Type', unique_types_1, default= unique_types_1, key='typecol8')
    with col7:
        st.markdown(f"#### Density Plot: {selected_column_1}")
        for date in selected_dates_1:
            for type in selected_types_1:
                part_data_1 = melted_data_1[(melted_data_1['Measurements'] == selected_column_1) & (melted_data_1['Date'] == date) & (melted_data_1['Part_Type'] == type)]['Data']
                sns.set(style="darkgrid")
                sns.distplot(part_data_1, hist=True, kde=True, bins=int(15), hist_kws={'edgecolor':'blue'},
                             kde_kws={'shade': True, 'linewidth': 3}, label=type + ' ' + date[5:])
    # Get the y-axis limits
    _, ymax_1 = plt.ylim()
    with col9:
        st.markdown("##### Legend and Text Parameters:")
        bbox_x_1 = st.slider('Legend x position: 1.25 ~ default', 0.0, 2.0, 1.25, key='bbox_x_1')
        bbox_y_1 = st.slider('Legend y position: 1.02 ~ default', 0.0, 2.0, 1.0, key= 'bbox_y_1')
        vertical_alignment_options_1 = ['top', 'bottom', 'baseline', 'center_baseline']
        horizontal_alignment_options_1 = ['center', 'left', 'right']
        va_1 = st.selectbox('Text vertical alignment: bottom ~ default', vertical_alignment_options, index=0, key='va_1')
        ha_1 = st.selectbox('Text horizontal alignment: left ~ default', horizontal_alignment_options, index=0, key='ha_1')
        text_y_1 = st.slider('Text y position:', 0.0, ymax_1*2, ymax_1, key='text_y_1')
        if st.button('Reset Chart Effects', key='legend Type chart'):
            bbox_x_1 = 1.25
            bbox_y_1 = 1.02
            va_1 = 'bottom'
            ha_1= 'left'
            text_y_1= ymax_1
    with col7:
        # Add vertical lines for USL and LSL
        plt.axvline(x=usl_1, color='r', linestyle='--', label='USL'+ ' ' + str(usl_1))
        plt.text(usl_1, text_y_1, 'USL: ' + str(usl_1), color='r', va=va_1, ha=ha_1)
        if lsl_1 != 0:
            plt.axvline(x=lsl_1, color='r', linestyle='--', label='LSL'+ ' '+ str(lsl_1))
            plt.text(lsl_1, text_y_1, 'LSL: ' + str(lsl_1), color='r', va=va_1, ha=ha_1)
        plt.legend(prop={'size': 10}, loc='upper right', bbox_to_anchor=(bbox_x_1, bbox_y_1)) #, title = 'Legend')
        plt.title(f'Density Plot')
        plt.xlabel('Data')
        plt.ylabel('Density')
        st.pyplot(plt)
        plt.clf()
    
