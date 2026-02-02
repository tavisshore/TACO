#!/bin/bash
# Helper script to run training experiments with dstack

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if dstack is installed
if ! command -v dstack &> /dev/null; then
    echo -e "${YELLOW}Warning: dstack is not installed. Install it with: pip install dstack${NC}"
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local config=$1
    echo -e "${BLUE}Starting experiment: ${config}${NC}"
    dstack run . -f "tasks/${config}.yaml"
}

# Function to list all available configs
list_configs() {
    echo -e "${GREEN}Available experiment configurations:${NC}"
    echo ""
    for config in tasks/*.yaml; do
        config_name=$(basename "$config" .yaml)
        echo "  - $config_name"
    done
    echo ""
}

# Main script logic
case "${1:-}" in
    "")
        echo "Usage: ./run_experiments.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  list              - List all available experiment configurations"
        echo "  run <config>      - Run a single experiment"
        echo "  run-all           - Run all experiments sequentially"
        echo "  baseline          - Run baseline experiment"
        echo "  status            - Show status of all runs"
        echo "  logs <run-name>   - Show logs for a specific run"
        echo ""
        echo "Examples:"
        echo "  ./run_experiments.sh list"
        echo "  ./run_experiments.sh run baseline"
        echo "  ./run_experiments.sh run-all"
        echo "  ./run_experiments.sh status"
        ;;

    "list")
        list_configs
        ;;

    "run")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a configuration name${NC}"
            list_configs
            exit 1
        fi

        if [ ! -f "tasks/$2.yaml" ]; then
            echo -e "${YELLOW}Error: Configuration 'tasks/$2.yaml' not found${NC}"
            list_configs
            exit 1
        fi

        run_experiment "$2"
        ;;

    "run-all")
        echo -e "${GREEN}Running all experiments...${NC}"
        for config in tasks/*.yaml; do
            config_name=$(basename "$config" .yaml)
            run_experiment "$config_name"
            sleep 2  # Small delay between submissions
        done
        echo -e "${GREEN}All experiments queued!${NC}"
        ;;

    "baseline")
        run_experiment "baseline"
        ;;

    "status")
        echo -e "${GREEN}Current dstack runs:${NC}"
        dstack ps
        ;;

    "logs")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a run name${NC}"
            echo "Use './run_experiments.sh status' to see available runs"
            exit 1
        fi
        dstack logs "$2"
        ;;

    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo "Run './run_experiments.sh' without arguments to see usage"
        exit 1
        ;;
esac
