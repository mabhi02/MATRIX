# MATRIX - Meta-learning Adaptive Thompson Reinforcement In X-space

A sophisticated AI system for medical diagnosis that combines dual-perspective neural agents with adaptive Thompson sampling for intelligent decision-making in clinical settings.

## 🚀 Overview

MATRIX is an advanced medical diagnosis system that employs a novel dual-agent architecture to simulate optimistic and pessimistic clinical perspectives. The system uses meta-learning, attention mechanisms, and Thompson sampling to adaptively improve diagnostic accuracy over time.

### Key Features

- **Dual-Agent Architecture**: Optimistic and pessimistic agents provide balanced clinical perspectives
- **Adaptive Thompson Sampling**: Dynamic exploration-exploitation balance for decision-making
- **State Space Encoding**: Advanced neural embedding of patient responses and medical context
- **Pattern Analysis**: Real-time analysis of medical text for risk assessment
- **Meta-Learning**: Continuous improvement through feedback-driven adaptation
- **Vector Database Integration**: Leverages medical knowledge bases for context-aware decisions

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

MATRIX employs a sophisticated multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    MATRIX Core System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │   Optimist  │    │   Pessimist  │    │  Thompson   │     │
│  │    Agent    │    │    Agent     │    │  Sampling   │     │
│  └─────────────┘    └──────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │   State     │    │   Pattern    │    │   Meta      │     │
│  │  Encoder    │    │   Analyzer   │    │  Learning   │     │
│  └─────────────┘    └──────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │   MATRIX    │    │    Vector    │    │   OpenAI    │     │
│  │   Decoder   │    │   Database   │    │ Embeddings  │     │
│  └─────────────┘    └──────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Dual-Agent System**: 
   - `OptimistAgent`: Focuses on positive indicators and less severe interpretations
   - `PessimistAgent`: Emphasizes warning signs and potential risks

2. **Thompson Sampling**: Adaptive decision-making with exploration-exploitation balance

3. **State Space Encoding**: Neural embedding of patient responses and medical context

4. **Pattern Analysis**: Real-time medical text analysis for risk assessment

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- OpenAI API key
- Pinecone API key
- Groq API key

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MATRIX
```

2. **Install dependencies**:
```bash
pip install torch torchvision
pip install openai pinecone-client groq
pip install python-dotenv numpy
```

3. **Environment Configuration**:
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

4. **Initialize Pinecone Index**:
Ensure you have a Pinecone index named `"who-guide-old"` with medical knowledge data.

## 🚀 Usage

### Basic Usage

```python
from matrix_core import MATRIX
from integration import MATRIXIntegration

# Initialize MATRIX system
matrix_integration = MATRIXIntegration()

# Run complete diagnosis workflow
result = matrix_integration.run_diagnosis()
```

### Command Line Interface

```bash
python run_matrix.py
```

This will start an interactive medical assessment session with:
1. Initial patient screening questions
2. MATRIX-powered follow-up questions
3. Examination recommendations
4. Diagnostic analysis

### Integration with Existing Systems

```python
from integration import MATRIXIntegration

# Initialize MATRIX
matrix = MATRIXIntegration()

# Process follow-up questions
should_continue = matrix.process_followup_question(
    initial_responses,
    followup_responses, 
    current_question
)

# Process examinations
should_examine = matrix.process_examination(
    initial_responses,
    followup_responses,
    exam_responses,
    current_exam
)
```

## 🔧 Components

### Core Modules

#### `matrix_core.py`
- **MATRIX**: Main system orchestrator
- **MetaLearningController**: Combines agent perspectives and manages learning

#### `agents.py`
- **OptimistAgent**: Positive-bias neural agent with attention mechanisms
- **PessimistAgent**: Risk-focused neural agent for comprehensive analysis

#### `state_encoder.py`
- **StateSpaceEncoder**: Converts patient responses into neural embeddings

#### `thompson_sampling.py`
- **AdaptiveThompsonSampler**: Implements Thompson sampling for agent selection

#### `pattern_analyzer.py`
- **PatternAnalyzer**: Analyzes medical text for severity and risk indicators
- **AdaptiveAgent**: Base class for pattern-aware agents

#### `decoder.py`
- **MATRIXDecoder**: Dual-path transformer decoder for agent outputs

#### `cmdML.py`
- Medical assessment workflow and user interface
- Vector database integration
- Response validation and processing

### Configuration

#### `config.py`
Central configuration management with parameters for:
- Model architecture (embedding dimensions, attention heads)
- Agent behavior (confidence thresholds, bias parameters)
- Learning rates and adaptation parameters
- System constraints and thresholds

## ⚙️ Configuration

Key configuration parameters in `MATRIXConfig`:

```python
# Model Architecture
EMBEDDING_DIM = 1536  # OpenAI embedding dimension
NUM_ATTENTION_HEADS = 8
DROPOUT_RATE = 0.1

# Agent Behavior
OPTIMIST_BASE_CONFIDENCE = 0.6
PESSIMIST_BASE_CONFIDENCE = 0.4
SIMILARITY_THRESHOLD = 2.8
EXAM_SIMILARITY_THRESHOLD = 1.5

# Learning Parameters
META_LEARNING_RATE = 0.01
CONFIDENCE_UPDATE_RATE = 0.1
MAX_QUESTIONS = 10
MAX_EXAMS = 5
```

## 📚 API Reference

### MATRIX Class

#### `process_state(initial_responses, followup_responses, current_question)`
Processes current patient state through dual-agent analysis.

**Parameters:**
- `initial_responses`: List of initial patient responses
- `followup_responses`: List of follow-up responses  
- `current_question`: Current question being evaluated

**Returns:**
- Dictionary with combined agent analysis and confidence scores

#### `update_from_feedback(success)`
Updates system based on feedback from diagnostic outcomes.

**Parameters:**
- `success`: Boolean indicating diagnostic success

### Agent Classes

#### `OptimistAgent.evaluate(state_embedding, initial_text)`
Evaluates patient state with optimistic bias.

#### `PessimistAgent.evaluate(state_embedding, initial_text)`
Evaluates patient state with pessimistic bias.

### Integration Class

#### `MATRIXIntegration.process_followup_question(initial_responses, followup_responses, current_question)`
Determines if a follow-up question should be asked.

## 💡 Examples

### Basic Medical Assessment

```python
from run_matrix import main

# Run complete assessment
main()
```

### Custom Agent Configuration

```python
from matrix_core import MATRIX
from config import MATRIXConfig

# Customize configuration
MATRIXConfig.SIMILARITY_THRESHOLD = 3.0
MATRIXConfig.OPTIMIST_BASE_CONFIDENCE = 0.7

# Initialize with custom config
matrix = MATRIX()
```

### Pattern Analysis

```python
from pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()
patterns = analyzer.analyze_patterns("severe chest pain for 3 hours")
print(f"Optimist confidence: {patterns['optimist_confidence']}")
print(f"Pessimist confidence: {patterns['pessimist_confidence']}")
```

## 🔍 Advanced Features

### Attention Visualization
```python
from attention_viz import visualize_attention

# Visualize agent attention patterns
visualize_attention(matrix.optimist, state_embedding)
```

### Decoder Tuning
```python
from decoder_tuner import tune_decoder

# Fine-tune decoder parameters
tune_decoder(matrix.meta_learner.decoder, training_data)
```

## 📊 Performance Monitoring

The system includes built-in monitoring for:
- Agent confidence levels
- Thompson sampling exploration rates
- Pattern analysis accuracy
- Diagnostic decision quality

## 🔬 Research Background

MATRIX implements cutting-edge research in:
- **Multi-Agent Reinforcement Learning**: Dual-agent architecture with competing perspectives
- **Thompson Sampling**: Bayesian optimization for decision-making under uncertainty
- **Meta-Learning**: Adaptive learning algorithms that improve over time
- **Medical AI**: Specialized neural architectures for clinical decision support

For detailed technical documentation and research findings, see the [comprehensive technical document](https://drive.google.com/file/d/1Tthkmz0aMIcVs_6fwQNNUyGvt4y5vxb4/view).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Type checking
mypy matrix_core.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for embedding models
- Pinecone for vector database services
- PyTorch team for the neural network framework
- Medical research community for domain knowledge

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the [detailed technical documentation](https://drive.google.com/file/d/1Tthkmz0aMIcVs_6fwQNNUyGvt4y5vxb4/view)
- Contact the development team

---

**MATRIX** - Advancing medical AI through adaptive multi-agent learning systems.