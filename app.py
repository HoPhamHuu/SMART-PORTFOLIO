
import streamlit as st
import pandas as pd
import numpy as np
from vnstock import Vnstock
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import random
import streamlit as st
import pandas as pd
import datetime
# Hàm tạo màu ngẫu nhiên
def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)


# Cấu hình giao diện trang web
st.set_page_config(page_title="Portfolio Optimization Dashboard 📈", layout="wide")
st.title("Portfolio Optimization Dashboard")
st.write("Ứng dụng tích hợp quy trình: tải dữ liệu cổ phiếu, xử lý, tối ưu hóa danh mục đầu tư (SLSQP, SGD, SGD - Sharpe), so sánh với VN-Index và trực quan hóa dữ liệu.")

# Tạo các tab ngang cho các trang
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Tải dữ liệu cổ phiếu",
    "Tối ưu danh mục (SLSQP)",
    "Tối ưu danh mục (SGD)",
    "Tối ưu danh mục (SGD - Sharpe)",
    "Trực quan hóa dữ liệu",
    "Thông tin công ty",
    "Báo cáo tài chính",
    "Phân tích kỹ thuật"
])

###########################################
# Tab 1: Tải dữ liệu cổ phiếu
###########################################
with tab1:
    st.header("Nhập mã cổ phiếu và tải dữ liệu")
    st.write("Nhập các mã cổ phiếu (phân cách bởi dấu phẩy, ví dụ: ACB, VCB):")
    symbols_input = st.text_input("Mã cổ phiếu")
    
    if st.button("Tải dữ liệu"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Danh sách mã cổ phiếu không được để trống!")
        else:
            # Lưu symbols vào session state
            st.session_state['symbols'] = symbols
            all_data = []
            for symbol in symbols:
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    historical_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                    if historical_data.empty:
                        st.warning(f"Không tìm thấy dữ liệu cho mã: {symbol}")
                        continue
                    historical_data['symbol'] = symbol
                    all_data.append(historical_data)
                    st.success(f"Đã tải dữ liệu cho: {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")
            if all_data:
                final_data = pd.concat(all_data, ignore_index=True)
                st.write("Đã kết hợp toàn bộ dữ liệu thành công!")
                
                def calculate_features(data):
                    data['daily_return'] = data['close'].pct_change()
                    data['volatility'] = data['daily_return'].rolling(window=30).std()
                    data.dropna(inplace=True)
                    return data
                
                processed_data = final_data.groupby('symbol').apply(calculate_features)
                processed_data = processed_data.reset_index(drop=True)
                processed_data.to_csv("processed_stock_data.csv", index=False)
                st.success("Dữ liệu xử lý đã được lưu vào file 'processed_stock_data.csv'.")
                st.dataframe(processed_data)
            else:
                st.error("Không có dữ liệu hợp lệ để xử lý!")

###########################################
# Tab 2: Tối ưu danh mục (SLSQP)
###########################################
with tab2:
    st.header("Tối ưu danh mục (SLSQP)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý thành công.")
    except FileNotFoundError:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu' trước.")
        st.stop()
    
    # Tính toán kỳ vọng lợi nhuận và ma trận hiệp phương sai
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def objective(weights, expected_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))
    total_expected_return = expected_returns.sum()
    init_weights = expected_returns / total_expected_return
    result = minimize(objective, init_weights, args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights_slsqp = result.x

    st.subheader("Trọng số tối ưu (SLSQP):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Cổ phiếu: {symbol}, Trọng số tối ưu: {optimal_weights_slsqp[i]:.4f}")
    
    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_slsqp = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_slsqp
    })
    portfolio_data_filtered = portfolio_data_slsqp[portfolio_data_slsqp['Trọng số tối ưu'] > 0]

    fig_slsqp = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Trọng số tối ưu (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_slsqp.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu đã lọc (nếu cần)
    fig_slsqp.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Trọng số tối ưu'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
        ),
        row=1, col=2
    )

    fig_slsqp.update_layout(
        title="So sánh trọng số tối ưu (SLSQP)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_slsqp, use_container_width=True)
    
    # Tính lợi nhuận tích lũy của danh mục
    processed_data['weighted_return_slsqp'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_slsqp))
    )
    portfolio_daily_return_slsqp = processed_data.groupby('time')['weighted_return_slsqp'].sum().reset_index()
    portfolio_daily_return_slsqp.rename(columns={'weighted_return_slsqp': 'daily_return'}, inplace=True)
    portfolio_daily_return_slsqp['cumulative_portfolio_return'] = (1 + portfolio_daily_return_slsqp['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SLSQP)")
    st.line_chart(portfolio_daily_return_slsqp.set_index('time')['cumulative_portfolio_return'])
    
    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()
        
        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
        
        comparison_slsqp = pd.merge(
            portfolio_daily_return_slsqp,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        comparison_slsqp.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (SLSQP)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index'
        }, inplace=True)
        
        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_slsqp[['time', 'Lợi nhuận danh mục (SLSQP)', 'Lợi nhuận VN-Index']].tail(10))
        
        fig_comp_slsqp = go.Figure()
        fig_comp_slsqp.add_trace(go.Scatter(
            x=comparison_slsqp['time'],
            y=comparison_slsqp['Lợi nhuận danh mục (SLSQP)'],
            mode='lines',
            name='Lợi nhuận danh mục (SLSQP)',
            line=dict(color='blue', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (SLSQP): %{y:.2%}<extra></extra>'
        ))
        fig_comp_slsqp.add_trace(go.Scatter(
            x=comparison_slsqp['time'],
            y=comparison_slsqp['Lợi nhuận VN-Index'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2%}<extra></extra>'
        ))
        fig_comp_slsqp.update_layout(
            title="So sánh lợi nhuận danh mục (SLSQP) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy",
            template="plotly_white"
        )
        st.plotly_chart(fig_comp_slsqp, use_container_width=True)
        comparison_slsqp.to_csv("portfolio_vs_vnindex_comparison_slsqp.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_slsqp.csv'.")

###########################################
# Tab 3: Tối ưu danh mục (SGD)
###########################################
with tab3:
    st.header("Tối ưu danh mục (SGD)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("Không tìm thấy file 'processed_stock_data.csv'. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()
    
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def return_based_weights(expected_returns):
        return expected_returns / expected_returns.sum()
    
    # Thêm hàm chiếu trọng số lên simplex để đảm bảo tổng trọng số = 1 và các trọng số không âm
    def project_simplex(v, s=1):
        v = np.maximum(v, 0)
        total = np.sum(v)
        return v / total * s if total != 0 else v
    
    def sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000):
        weights = return_based_weights(expected_returns)
        for epoch in range(epochs):
            grad = np.dot(cov_matrix, weights) / portfolio_volatility(weights, cov_matrix)
            weights -= learning_rate * grad
            # Sử dụng project_simplex để đảm bảo ràng buộc
            weights = project_simplex(weights)
        return weights
    
    optimal_weights_sgd = sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000)
    
    st.subheader("Trọng số tối ưu (SGD):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Cổ phiếu: {symbol}, Trọng số tối ưu: {optimal_weights_sgd[i]:.4f}")
    
    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_sgd = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_sgd
    })
    portfolio_data_filtered = portfolio_data_sgd[portfolio_data_sgd['Trọng số tối ưu'] > 0]

    fig_sgd = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Trọng số tối ưu (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_sgd.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu đã lọc
    fig_sgd.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Trọng số tối ưu'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
        ),
        row=1, col=2
    )

    fig_sgd.update_layout(
        title="So sánh trọng số tối ưu (SGD)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_sgd, use_container_width=True)

    # Tính lợi nhuận tích lũy của danh mục
    processed_data['weighted_return_sgd'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd))
    )
    portfolio_daily_return_sgd = processed_data.groupby('time')['weighted_return_sgd'].sum().reset_index()
    portfolio_daily_return_sgd.rename(columns={'weighted_return_sgd': 'daily_return'}, inplace=True)
    portfolio_daily_return_sgd['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sgd['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SGD)")
    st.line_chart(portfolio_daily_return_sgd.set_index('time')['cumulative_portfolio_return'])
    
    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()
        
        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
        
        comparison_sgd = pd.merge(
            portfolio_daily_return_sgd,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        comparison_sgd.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (SGD)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index'
        }, inplace=True)
        
        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_sgd[['time', 'Lợi nhuận danh mục (SGD)', 'Lợi nhuận VN-Index']].tail(10))
        
        fig_comp_sgd = go.Figure()
        fig_comp_sgd.add_trace(go.Scatter(
            x=comparison_sgd['time'],
            y=comparison_sgd['Lợi nhuận danh mục (SGD)'],
            mode='lines',
            name='Lợi nhuận danh mục (SGD)',
            line=dict(color='green', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (SGD): %{y:.2%}<extra></extra>'
        ))
        fig_comp_sgd.add_trace(go.Scatter(
            x=comparison_sgd['time'],
            y=comparison_sgd['Lợi nhuận VN-Index'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2%}<extra></extra>'
        ))
        fig_comp_sgd.update_layout(
            title="So sánh lợi nhuận danh mục (SGD) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp_sgd, use_container_width=True)
        comparison_sgd.to_csv("portfolio_vs_vnindex_comparison_sgd.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_sgd.csv'.")

###########################################
# Tab 4: Tối ưu danh mục (SGD - Sharpe)
###########################################
with tab4:
    st.header("Tối ưu danh mục (SGD - Sharpe)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()

    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    cov_matrix = processed_data.pivot(index='time', columns='symbol', values='daily_return').cov()

    # Thêm hàm chiếu trọng số lên simplex
    def project_simplex(v, s=1):
        v = np.maximum(v, 0)
        total = np.sum(v)
        return v / total * s if total != 0 else v

    def sgd_portfolio_optimization_sharpe(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000):
        weights = (expected_returns / expected_returns.sum())
        weights = weights.values  # chuyển sang mảng NumPy để thuận tiện tính toán
        for epoch in range(epochs):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            grad = 2 * np.dot(cov_matrix, weights)
            weights -= learning_rate * grad
            # Sử dụng project_simplex để đảm bảo trọng số không âm và tổng = 1
            weights = project_simplex(weights)
        return weights

    optimal_weights_sgd_sharpe = sgd_portfolio_optimization_sharpe(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000)

    st.subheader("Trọng số tối ưu (SGD - Sharpe):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Cổ phiếu: {symbol}, Trọng số tối ưu: {optimal_weights_sgd_sharpe[i]:.4f}")
    sharpe_ratio = np.dot(optimal_weights_sgd_sharpe, expected_returns) / np.sqrt(np.dot(optimal_weights_sgd_sharpe.T, np.dot(cov_matrix, optimal_weights_sgd_sharpe)))
    st.write(f"Tỷ lệ Sharpe tốt nhất: {sharpe_ratio:.4f}")
    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_sharpe = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_sgd_sharpe
    })
    portfolio_data_filtered = portfolio_data_sharpe[portfolio_data_sharpe['Trọng số tối ưu'] > 0]

    fig_sharpe = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Trọng số tối ưu (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_sharpe.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu đã lọc
    fig_sharpe.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Trọng số tối ưu'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
        ),
        row=1, col=2
    )

    fig_sharpe.update_layout(
        title="So sánh trọng số tối ưu (SGD - Sharpe)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)
    
    # Tính lợi nhuận tích lũy của danh mục
    processed_data['weighted_return_sharpe'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd_sharpe))
    )
    portfolio_daily_return_sharpe = processed_data.groupby('time')['weighted_return_sharpe'].sum().reset_index()
    portfolio_daily_return_sharpe.rename(columns={'weighted_return_sharpe': 'daily_return'}, inplace=True)
    portfolio_daily_return_sharpe['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sharpe['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SGD - Sharpe)")
    st.line_chart(portfolio_daily_return_sharpe.set_index('time')['cumulative_portfolio_return'])
    
    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()
        
        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
        
        comparison_sharpe = pd.merge(
            portfolio_daily_return_sharpe,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        comparison_sharpe.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (Sharpe)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index'
        }, inplace=True)
        
        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_sharpe[['time', 'Lợi nhuận danh mục (Sharpe)', 'Lợi nhuận VN-Index']].tail(10))
        
        fig_comp_sharpe = go.Figure()
        fig_comp_sharpe.add_trace(go.Scatter(
            x=comparison_sharpe['time'],
            y=comparison_sharpe['Lợi nhuận danh mục (Sharpe)'],
            mode='lines',
            name='Lợi nhuận danh mục (Sharpe)',
            line=dict(color='orange', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (Sharpe): %{y:.2%}<extra></extra>'
        ))
        fig_comp_sharpe.add_trace(go.Scatter(
            x=comparison_sharpe['time'],
            y=comparison_sharpe['Lợi nhuận VN-Index'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2%}<extra></extra>'
        ))
        fig_comp_sharpe.update_layout(
            title="So sánh lợi nhuận danh mục (Sharpe) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp_sharpe, use_container_width=True)
        comparison_sharpe.to_csv("portfolio_vs_vnindex_comparison_sharpe.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_sharpe.csv'.")

###########################################
# Tab 5: Trực quan hóa dữ liệu
###########################################
with tab5:
    st.header("Trực quan hóa dữ liệu")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
    except Exception as e:
        st.error("Không thể tải file 'processed_stock_data.csv'. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()
    
    st.subheader("Xu hướng giá đóng cửa cổ phiếu theo thời gian")
    fig1 = px.line(
        processed_data,
        x='time',
        y='close',
        color='symbol',
        title='Xu hướng giá đóng cửa cổ phiếu theo thời gian',
        labels={'time': 'Thời gian', 'close': 'Giá đóng cửa', 'symbol': 'Mã cổ phiếu'},
    )
    fig1.update_layout(
        xaxis_title='Thời gian',
        yaxis_title='Giá đóng cửa',
        legend_title='Mã cổ phiếu',
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Biểu đồ nhiệt tương quan giá đóng cửa")
    close_data = processed_data.pivot_table(values='close', index='time', columns='symbol')
    correlation_matrix = close_data.corr()
    rounded_correlation = correlation_matrix.round(2)
    fig2 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        colorbar=dict(title='Hệ số tương quan'),
    ))
    for i in range(len(rounded_correlation)):
        for j in range(len(rounded_correlation.columns)):
            fig2.add_annotation(
                text=str(rounded_correlation.iloc[i, j]),
                x=rounded_correlation.columns[j],
                y=rounded_correlation.index[i],
                showarrow=False,
                font=dict(color='black' if rounded_correlation.iloc[i, j] < 0 else 'white')
            )
    fig2.update_traces(
        hovertemplate='<b>Mã cổ phiếu: %{x}</b><br>' +
                      '<b>Mã cổ phiếu: %{y}</b><br>' +
                      'Hệ số tương quan: %{z:.4f}<extra></extra>'
    )
    fig2.update_layout(
        title='Biểu đồ nhiệt tương quan giá đóng cửa',
        xaxis_title='Mã cổ phiếu',
        yaxis_title='Mã cổ phiếu'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Biểu đồ nhiệt tương quan lợi nhuận hàng ngày")
    returns_data = processed_data.pivot_table(index='time', columns='symbol', values='daily_return')
    correlation_matrix_returns = returns_data.corr()
    fig3 = ff.create_annotated_heatmap(
        z=correlation_matrix_returns.values,
        x=correlation_matrix_returns.columns.tolist(),
        y=correlation_matrix_returns.columns.tolist(),
        colorscale='RdBu',
        zmin=-1, zmax=1
    )
    fig3.update_layout(title="Ma trận tương quan giữa các cổ phiếu")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Biến động cổ phiếu theo thời gian")
    fig4 = px.line(processed_data, x='time', y='volatility', color='symbol', title="Biến động cổ phiếu theo thời gian")
    fig4.update_xaxes(title_text='Ngày')
    fig4.update_yaxes(title_text='Biến động')
    st.plotly_chart(fig4, use_container_width=True)

###########################################
# Tab 6: Thông tin công ty
###########################################
###########################################
# Tab 6: Thông tin công ty
###########################################
with tab6:
    st.header("Thông tin tổng hợp về các công ty")
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở tab 'Tải dữ liệu cổ phiếu' trước.")
    else:
        symbols = st.session_state['symbols']
        
        for symbol in symbols:
            st.subheader(f"Thông tin cho mã {symbol}")
            try:
                company = Vnstock().stock(symbol=symbol, source='TCBS').company
                
                with st.expander("**Hồ sơ công ty:**"):
                    profile = company.profile()
                    if isinstance(profile, pd.DataFrame):
                        st.dataframe(profile)
                    else:
                        st.write(profile)
                
                with st.expander("**Cổ đông:**"):
                    shareholders = company.shareholders()
                    if isinstance(shareholders, pd.DataFrame):
                        st.dataframe(shareholders)
                    else:
                        st.write(shareholders)
                
                with st.expander("**Giao dịch nội bộ:**"):
                    insider_deals = company.insider_deals()
                    if isinstance(insider_deals, pd.DataFrame):
                        st.dataframe(insider_deals)
                    else:
                        st.write(insider_deals)
                
                with st.expander("**Công ty con:**"):
                    subsidiaries = company.subsidiaries()
                    if isinstance(subsidiaries, pd.DataFrame):
                        st.dataframe(subsidiaries)
                    else:
                        st.write(subsidiaries)
                
                with st.expander("**Ban điều hành:**"):
                    officers = company.officers()
                    if isinstance(officers, pd.DataFrame):
                        st.dataframe(officers)
                    else:
                        st.write(officers)
                
                with st.expander("**Sự kiện:**"):
                    events = company.events()
                    if isinstance(events, pd.DataFrame):
                        st.dataframe(events)
                    else:
                        st.write(events)
                
                with st.expander("**Tin tức:**"):
                    news = company.news()
                    if isinstance(news, list) and all(isinstance(item, dict) for item in news):
                        for item in news:
                            st.write(f"- {item.get('title', 'N/A')} ({item.get('date', 'N/A')})")
                            st.write(item.get('summary', 'Không có tóm tắt'))
                            url = item.get('url', None)
                            if url:
                                st.write(f"[Đọc thêm]({url})")
                            else:
                                st.write("Không có URL")
                    else:
                        st.write("Tin tức không khả dụng hoặc định dạng không đúng:")
                        st.write(news)
                
                with st.expander("**Cổ tức:**"):
                    dividends = company.dividends()
                    if isinstance(dividends, pd.DataFrame):
                        st.dataframe(dividends)
                    else:
                        st.write(dividends)
            
            except Exception as e:
                st.error(f"Lỗi khi tải thông tin cho mã {symbol}: {e}")

###########################################
# Tab 7: Báo cáo tài chính
###########################################
with tab7:
    st.header("Tổng hợp báo cáo tài chính")
    
    # Cấu hình Plotly: modebar luôn hiển thị
    config = {
        "displayModeBar": True,
        "displaylogo": False
    }
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở trang 'Fetch Stock Data' trước.")
    else:
        symbols = st.session_state['symbols']

        def rename_duplicate_columns(df):
            if df.empty:
                return df
            if isinstance(df.columns, pd.MultiIndex):
                flat_columns = [
                    '_'.join(str(col).strip() for col in multi_col if str(col).strip())
                    for multi_col in df.columns
                ]
            else:
                flat_columns = df.columns.tolist()
            seen = {}
            final_columns = []
            for col in flat_columns:
                if col in seen:
                    seen[col] += 1
                    final_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_columns.append(col)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.Index(final_columns)
            else:
                df.columns = final_columns
            return df

        # CSS cho nội dung của expander (background trắng, đổ bóng,...)
        st.markdown(
            """
            <style>
            .streamlit-expanderContent {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Hàm random_color (nếu chưa có)
        import random
        def random_color():
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            return random.choice(colors)

        # --- Sử dụng caching để tải dữ liệu tài chính ---
        @st.cache_data(show_spinner=False)
        def get_financial_data(symbol, report_type):
            try:
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                if report_type == "balance":
                    data = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
                elif report_type == "income":
                    data = stock.finance.income_statement(period='year', lang='vi', dropna=True)
                elif report_type == "cashflow":
                    data = stock.finance.cash_flow(period='year', lang="vi", dropna=True)
                elif report_type == "ratios":
                    data = stock.finance.ratio(period='year', lang='vi', dropna=True)
                else:
                    data = pd.DataFrame()
                data = rename_duplicate_columns(data)
                return data
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu cho {symbol} - {report_type}: {e}")
                return pd.DataFrame()
        
        # ----------------------- Lặp qua từng mã cổ phiếu -----------------------
        for symbol in symbols:
            st.header(f"Báo cáo tài chính cho mã {symbol}")

            # ----------------------- BALANCE SHEET -----------------------
            with st.expander("Bảng cân đối kế toán (Hàng năm)"):
                balance_data = get_financial_data(symbol, "balance")
                if not balance_data.empty and 'Năm' in balance_data.columns:
                    st.write("**Bảng cân đối kế toán (Hàng năm):**")
                    st.dataframe(balance_data)
                    numeric_cols = [col for col in balance_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Bảng cân đối {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(balance_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Bảng cân đối {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = balance_data[balance_data['Năm'].isin(selected_years)] if selected_years else balance_data

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_bar")
                                        
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho bảng cân đối kế toán của {symbol}")

            # ----------------------- INCOME STATEMENT -----------------------
            with st.expander("Báo cáo lãi lỗ (Hàng năm)"):
                income_data = get_financial_data(symbol, "income")
                if not income_data.empty and 'Năm' in income_data.columns:
                    st.write("**Báo cáo lãi lỗ (Hàng năm):**")
                    st.dataframe(income_data)
                    numeric_cols = [col for col in income_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lãi lỗ {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(income_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Báo cáo lãi lỗ {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = income_data[income_data['Năm'].isin(selected_years)] if selected_years else income_data

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"income_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"income_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lãi lỗ của {symbol}")

            # ----------------------- CASH FLOW -----------------------
            with st.expander("Báo cáo lưu chuyển tiền tệ (Hàng năm)"):
                cash_flow_data = get_financial_data(symbol, "cashflow")
                if not cash_flow_data.empty and 'Năm' in cash_flow_data.columns:
                    st.write("**Báo cáo lưu chuyển tiền tệ (Hàng năm):**")
                    st.dataframe(cash_flow_data)
                    numeric_cols = [col for col in cash_flow_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lưu chuyển {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(cash_flow_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Báo cáo lưu chuyển {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = cash_flow_data[cash_flow_data['Năm'].isin(selected_years)] if selected_years else cash_flow_data

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lưu chuyển tiền tệ của {symbol}")

            # ----------------------- FINANCIAL RATIOS -----------------------
            with st.expander("Chỉ số tài chính (Hàng năm)"):
                ratios_data = get_financial_data(symbol, "ratios")
                if not ratios_data.empty and 'Meta_Năm' in ratios_data.columns:
                    st.write("**Chỉ số tài chính (Hàng năm):**")
                    st.dataframe(ratios_data)
                    numeric_cols = [col for col in ratios_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Meta_Năm']
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Chỉ số tài chính {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(ratios_data['Meta_Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Chỉ số tài chính {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = ratios_data[ratios_data['Meta_Năm'].isin(selected_years)] if selected_years else ratios_data

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Meta_Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Meta_Năm')
                                                start_year = df_sorted['Meta_Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Meta_Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Meta_Năm' cho chỉ số tài chính của {symbol}")
# Phân tích kỹ thuật
with tab8:
    st.header("Phân tích kỹ thuật")

    # **Chọn mã cổ phiếu**
    stock_symbol = st.text_input("Nhập mã cổ phiếu (ví dụ: VCI)", value="VCI").upper()

    # **Chọn khoảng thời gian**
    start_date = st.date_input("Chọn ngày bắt đầu", value=datetime.datetime(2020, 1, 1))
    end_date = st.date_input("Chọn ngày kết thúc", value=datetime.datetime.now())

    # **Lấy dữ liệu từ vnstock**
    try:
        stock = Vnstock().stock(symbol=stock_symbol, source='VCI')
        stock_data = stock.quote.history(start=start_date.strftime('%Y-%m-%d'),
                                         end=end_date.strftime('%Y-%m-%d'))
        if stock_data.empty:
            st.error(f"Không có dữ liệu cho mã {stock_symbol} trong khoảng thời gian đã chọn.")
            st.stop()
        stock_data['time'] = pd.to_datetime(stock_data['time'])
        stock_data = stock_data.sort_values('time')
        st.success(f"Đã tải dữ liệu cho mã {stock_symbol} từ {start_date} đến {end_date}.")
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.stop()

    # **Chọn chỉ báo kỹ thuật**
    indicators = st.multiselect(
        "Chọn chỉ báo kỹ thuật",
        [
            "SMA (Đường trung bình động đơn giản)", 
            "EMA (Đường trung bình động hàm mũ)", 
            "RSI (Chỉ số sức mạnh tương đối)", 
            "MACD", 
            "Bollinger Bands",
            "Stochastic Oscillator",
            "CCI (Commodity Channel Index)",
            "ADX (Average Directional Index)",
            "DMI"
        ]
    )

    # Nhập khoảng thời gian cho các chỉ báo nếu được chọn
    if "SMA (Đường trung bình động đơn giản)" in indicators:
        sma_period = st.number_input("Chọn khoảng thời gian cho SMA", min_value=1, max_value=200, value=50)
    if "EMA (Đường trung bình động hàm mũ)" in indicators:
        ema_period = st.number_input("Chọn khoảng thời gian cho EMA", min_value=1, max_value=200, value=50)
    if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
        rsi_period = st.number_input("Chọn khoảng thời gian cho RSI", min_value=1, max_value=100, value=14)
    if "Bollinger Bands" in indicators:
        bb_period = st.number_input("Chọn khoảng thời gian cho Bollinger Bands", min_value=1, max_value=200, value=20)
    if "Stochastic Oscillator" in indicators:
        stoch_period = st.number_input("Chọn khoảng thời gian cho Stochastic Oscillator", min_value=1, max_value=100, value=14)
    if "CCI (Commodity Channel Index)" in indicators:
        cci_period = st.number_input("Chọn khoảng thời gian cho CCI", min_value=1, max_value=200, value=20)
    if "ADX (Average Directional Index)" in indicators:
        adx_period = st.number_input("Chọn khoảng thời gian cho ADX", min_value=1, max_value=100, value=14)
    if "DMI" in indicators:
        dmi_period = st.number_input("Chọn khoảng thời gian cho DMI", min_value=1, max_value=100, value=14)

    # Hàm tính toán các chỉ báo kỹ thuật
    def compute_indicators():
        global stock_data
        if "SMA (Đường trung bình động đơn giản)" in indicators:
            stock_data['SMA'] = stock_data['close'].rolling(window=sma_period).mean()
        if "EMA (Đường trung bình động hàm mũ)" in indicators:
            stock_data['EMA'] = stock_data['close'].ewm(span=ema_period, adjust=False).mean()
        if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))
        if "MACD" in indicators:
            stock_data['EMA12'] = stock_data['close'].ewm(span=12, adjust=False).mean()
            stock_data['EMA26'] = stock_data['close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
            stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        if "Bollinger Bands" in indicators:
            stock_data['Middle_Band'] = stock_data['close'].rolling(window=bb_period).mean()
            stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data['close'].rolling(window=bb_period).std()
            stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data['close'].rolling(window=bb_period).std()
        if "Stochastic Oscillator" in indicators:
            low_min = stock_data['low'].rolling(window=stoch_period).min()
            high_max = stock_data['high'].rolling(window=stoch_period).max()
            stock_data['%K'] = (stock_data['close'] - low_min) / (high_max - low_min) * 100
            stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
        if "CCI (Commodity Channel Index)" in indicators:
            tp = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3
            sma_tp = tp.rolling(window=cci_period).mean()
            mad = tp.rolling(window=cci_period).apply(lambda x: np.fabs(x - x.mean()).mean())
            stock_data['CCI'] = (tp - sma_tp) / (0.015 * mad)
        if "ADX (Average Directional Index)" in indicators:
            high = stock_data['high']
            low = stock_data['low']
            close = stock_data['close']
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=adx_period).mean()
            up_move = high - high.shift()
            down_move = low.shift() - low
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=adx_period).sum() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=adx_period).sum() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            stock_data['ADX'] = dx.rolling(window=adx_period).mean()
        if "DMI" in indicators:
            high = stock_data['high']
            low = stock_data['low']
            close = stock_data['close']
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=dmi_period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=dmi_period).sum() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=dmi_period).sum() / atr)
            stock_data['+DI'] = plus_di
            stock_data['-DI'] = minus_di

    # Chạy tính toán chỉ báo trong một tiến trình riêng
    indicator_thread = threading.Thread(target=compute_indicators)
    indicator_thread.start()
    indicator_thread.join()  # Chờ tiến trình tính toán hoàn thành

    # **Tạo biểu đồ với khối lượng có trục Y phụ**
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{}]])

    # **Thêm biểu đồ nến vào hàng trên (trục Y chính)**
    fig.add_trace(go.Candlestick(
        x=stock_data['time'],
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        name="Nến"
    ), row=1, col=1, secondary_y=False)

    # **Thêm khối lượng giao dịch vào trục Y phụ**
    fig.add_trace(go.Bar(
        x=stock_data['time'],
        y=stock_data['volume'],
        name="Khối lượng",
        marker_color='blue',
        opacity=0.4
    ), row=1, col=1, secondary_y=True)

    # **Thêm các chỉ báo kỹ thuật vào biểu đồ hàng trên**
    if "SMA (Đường trung bình động đơn giản)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['SMA'], 
                                 name=f"SMA {sma_period}", line=dict(color='orange')),
                      row=1, col=1, secondary_y=False)
    if "EMA (Đường trung bình động hàm mũ)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['EMA'], 
                                 name=f"EMA {ema_period}", line=dict(color='green')),
                      row=1, col=1, secondary_y=False)
    if "Bollinger Bands" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Upper_Band'], 
                                 name="Upper Band", line=dict(color='red')),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Middle_Band'], 
                                 name="Middle Band", line=dict(color='purple')),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Lower_Band'], 
                                 name="Lower Band", line=dict(color='red')),
                      row=1, col=1, secondary_y=False)

    # **Thêm các chỉ báo kỹ thuật vào hàng dưới**
    if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['RSI'], 
                                 name="RSI", line=dict(color='purple')),
                      row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    if "MACD" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['MACD'], 
                                 name="MACD", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Signal_Line'], 
                                 name="Signal Line", line=dict(color='red')),
                      row=2, col=1)
    if "Stochastic Oscillator" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['%K'], 
                                 name="Stochastic %K", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['%D'], 
                                 name="Stochastic %D", line=dict(color='orange')),
                      row=2, col=1)
    if "CCI (Commodity Channel Index)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['CCI'], 
                                 name="CCI", line=dict(color='brown')),
                      row=2, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
    if "ADX (Average Directional Index)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['ADX'], 
                                 name="ADX", line=dict(color='magenta')),
                      row=2, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=2, col=1)
    if "DMI" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['+DI'], 
                                 name="+DI", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['-DI'], 
                                 name="-DI", line=dict(color='red')),
                      row=2, col=1)

    # **Cập nhật giao diện**
    fig.update_layout(
        title=f"Phân tích kỹ thuật cho {stock_symbol} từ {start_date} đến {end_date}",
        height=800,
        showlegend=True,
        xaxis_title="Thời gian",
        yaxis_title="Giá",
        yaxis2=dict(title="Khối lượng", overlaying="y", side="right"),
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    # **Hiển thị biểu đồ**
    st.plotly_chart(fig, use_container_width=True)
