# 🔋 Battery Analytics Dashboard

A comprehensive, modular battery analytics dashboard built with Python, Panel, and MLflow for advanced battery design and data analytics workflows.

## 🎯 Project Overview

The Battery Analytics Dashboard is an end-to-end solution for battery research and development teams, providing interactive data visualization, statistical analysis, and machine learning capabilities for battery cycle data. The application connects to PostgreSQL databases via Redash API, processes large-scale battery test data, and provides actionable insights for battery design optimization.

### Key Features

- **📊 Interactive Data Visualization**: Multi-cell comparison, cycle analysis, and capacity fade tracking
- **🔍 Advanced Filtering**: Dynamic cell selection with metadata-based filtering
- **📈 Real-time Analytics**: Live data processing with caching for performance
- **🤖 ML Integration**: MLflow-powered machine learning workflows for predictive analytics
- **🗄️ Database Integration**: Direct PostgreSQL connectivity with optimized queries
- **⚡ High Performance**: Polars-based data processing for large datasets
- **🎨 Modern UI**: Panel-based responsive web interface with dark/light themes

## 🏗️ Architecture

The project follows a modular architecture designed from the end goal to implementation:

```
battery_analytics_dashboard/
├── battery_dashboard/
│   ├── core/                 # Core data management and state
│   ├── data/                 # Data loading and processing
│   ├── ui/                   # User interface components
│   ├── analytics/            # Battery-specific analytics
│   ├── database/             # Database models and queries
│   └── utils/                # Utilities and helpers
├── tests/                    # Comprehensive test suite
├── docs/                     # Documentation
└── deployment/               # Docker and K8s configurations
```

### Core Components

- **Data Managers**: `CycleDataManager` and `CellDataManager` for efficient data handling
- **State Management**: Global application state with reactive updates
- **Analytics Engine**: Battery-specific calculations and ML feature engineering
- **Visualization Layer**: Configurable plots with HoloViews and Plotly integration
- **MLflow Integration**: Experiment tracking and model deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL database with battery test data
- Redash instance (optional, for API-based data access)
- MLflow server (optional, for ML workflows)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourorg/battery-analytics-dashboard.git
   cd battery-analytics-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   # or for development
   pip install -e ".[dev]"
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env file with your database and API configurations
   ```

4. **Run the dashboard**:
   ```bash
   python run.py
   # or
   panel serve battery_dashboard/main.py --port 8060 --show
   ```

The dashboard will be available at `http://localhost:8060`

## ⚙️ Configuration

### Environment Variables

```bash
# Database Configuration
REDASH_URL=http://192.168.80.30:8080
REDASH_API_KEY=your_api_key_here
CELL_QUERY_ID=24
CYCLE_QUERY_ID=28

# PostgreSQL Database (direct connection)
DATABASE_URL=postgresql://username:password@localhost:5432/battery_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=battery_db
DB_USER=cell_admin
DB_PASSWORD=your_password

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=battery_analytics

# Application Configuration
LOG_LEVEL=INFO
DEBUG=False
CACHE_TTL=900
PANEL_PORT=8060
```

### Database Schema

The application expects a PostgreSQL database with the following key tables:
- `cell`: Battery cell metadata and specifications
- `mergedtest`: Test data consolidation
- `testfiles`: Individual test file records
- `cycle_data`: Processed cycle data (via Redash queries)

See `deployment/cell_process_and_test_db.sql` for complete schema.

## 📊 Usage

### Cell Selection and Filtering

1. **Navigate to Cell Selector tab**
2. **Apply filters** by design name, experiment group, layer types, test status, or year
3. **Select cells** for analysis using the interactive table
4. **View metadata** including cell specifications and test parameters

### Cycle Analysis

1. **Switch to Cycle Analysis tab**
2. **Configure plots** using the settings panel:
   - Choose X/Y axis variables
   - Select normalization methods
   - Configure series grouping
3. **Analyze trends** in capacity, energy, and efficiency
4. **Export data** for further analysis

### Statistical Analysis (Coming Soon)

- Descriptive statistics
- Correlation analysis
- Degradation modeling
- Comparative analysis

### ML Analysis (Coming Soon)

- Feature engineering
- Predictive modeling
- Anomaly detection
- Performance forecasting

## 🐳 Docker Deployment

### Single Container

```bash
docker build -t battery-dashboard .
docker run -p 8060:8060 --env-file .env battery-dashboard
```

### Full Stack with Docker Compose

```bash
docker-compose up -d
```

This launches:
- Battery Dashboard (port 8060)
- PostgreSQL database (port 5432)
- MLflow server (port 5000)
- MinIO storage (port 9000)
- Redis cache (port 6379)

## 🔬 Data Processing Pipeline

### Data Flow

1. **Data Ingestion**: Raw test data from battery cyclers → PostgreSQL
2. **Data Processing**: Redash queries → Polars DataFrames → Normalized features
3. **Caching**: Processed data cached for performance
4. **Analytics**: Real-time calculations and ML feature engineering
5. **Visualization**: Interactive plots and dashboards

### Key Data Processing Features

- **Polars Integration**: High-performance DataFrame operations
- **Intelligent Caching**: TTL-based caching with configurable invalidation
- **Data Validation**: Schema validation and data quality checks
- **Parallel Processing**: Multi-threaded data loading and processing
- **Error Handling**: Comprehensive error handling and logging

## 🧪 Machine Learning Workflows

### MLflow Integration

- **Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Version control for trained models
- **Feature Store**: Centralized feature engineering and storage
- **Deployment**: Model serving and batch prediction

### Feature Engineering

- Capacity fade analysis
- Rolling statistics (5, 10, 20 cycle windows)
- Temperature compensation
- Coulombic efficiency calculations
- Cycle-based features (progression, time since start)

### Model Types

- **Regression**: Capacity prediction, end-of-life forecasting
- **Classification**: Failure mode detection, quality classification
- **Time Series**: Degradation trend analysis
- **Anomaly Detection**: Outlier identification

## 🧪 Testing

Run the test suite:

```bash
pytest
# or with coverage
pytest --cov=battery_dashboard
```

Test categories:
- Unit tests for data processing functions
- Integration tests for database connections
- UI component tests
- End-to-end workflow tests

## 🛠️ Development

### Development Setup

```bash
git clone https://github.com/yourorg/battery-analytics-dashboard.git
cd battery-analytics-dashboard
pip install -e ".[dev]"
pre-commit install
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality assurance

### Architecture Principles

1. **Modular Design**: Separate concerns with clear interfaces
2. **Data-Driven**: Polars for high-performance data processing
3. **Reactive UI**: Panel widgets with automatic updates
4. **Configurable**: Environment-based configuration
5. **Observable**: Comprehensive logging and monitoring
6. **Testable**: Dependency injection and mock-friendly design

## 📖 API Documentation

Key modules and classes:

### Core Data Management
- `CycleDataManager`: Handles cycle data operations
- `CellDataManager`: Manages cell metadata
- `StateManager`: Application state management

### Data Processing
- `DataProcessor`: Main data transformation pipeline
- `RedashClient`: API client for data retrieval
- `FeatureStore`: ML feature engineering

### UI Components
- `CellSelectorTab`: Cell selection interface
- `CyclePlotsTab`: Visualization components
- `BaseTab`: Abstract base for UI tabs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📋 Roadmap

### Phase 1 (Current)
- ✅ Core dashboard functionality
- ✅ Cell selection and filtering
- ✅ Basic cycle analysis plots
- ✅ Database integration

### Phase 2 (In Progress)
- 🔄 Advanced statistical analysis
- 🔄 ML model integration
- 🔄 Enhanced data processing pipeline
- 🔄 Performance optimizations

### Phase 3 (Planned)
- 📅 Real-time data streaming
- 📅 Advanced ML workflows
- 📅 Multi-user support
- 📅 API endpoints for external integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Battery Analytics Team**
- Lead Developer: [Your Name]
- Data Scientists: [Team Members]
- Domain Experts: [Battery Researchers]

## 📞 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Email**: analytics@example.com

## 🙏 Acknowledgments

- Panel and HoloViews teams for the visualization framework
- Polars team for high-performance data processing
- MLflow team for experiment tracking capabilities
- Battery research community for domain expertise

---

**Built with ❤️ for the battery research community**

For issues, inquiries, or contributions, contact:

- **Name**: Srinath Chakravarthy
- **Email**: sc@factorialenergy.com