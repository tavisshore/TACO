#!/bin/bash
# Helper script to run training experiments with dstack (new format)

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
    echo -e "${BLUE}Applying task configuration: ${config}${NC}"
    dstack apply -f "tasks/${config}.dstack.yml"
}

# Function to list all available configs
list_configs() {
    echo -e "${GREEN}Available task configurations:${NC}"
    echo ""
    for config in tasks/*.dstack.yml; do
        config_name=$(basename "$config" .dstack.yml)
        task_name=$(grep "^name:" "$config" | head -1 | awk '{print $2}')
        echo "  - $config_name ($task_name)"
    done
    echo ""
}

# Main script logic
case "${1:-}" in
    "")
        echo "Usage: ./run_dstack.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  list              - List all available task configurations"
        echo "  apply <config>    - Apply a task configuration"
        echo "  run <config>      - Apply and run a task"
        echo "  status            - Show status of all runs"
        echo "  logs <run-name>   - Show logs for a specific run"
        echo "  stop <run-name>   - Stop a running task"
        echo "  delete <run-name> - Delete a run"
        echo ""
        echo "Examples:"
        echo "  ./run_dstack.sh list"
        echo "  ./run_dstack.sh apply baseline"
        echo "  ./run_dstack.sh run auto-batch-single-gpu"
        echo "  ./run_dstack.sh status"
        ;;

    "list")
        list_configs
        ;;

    "apply")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a configuration name${NC}"
            list_configs
            exit 1
        fi

        if [ ! -f "tasks/$2.dstack.yml" ]; then
            echo -e "${YELLOW}Error: Configuration 'tasks/$2.dstack.yml' not found${NC}"
            list_configs
            exit 1
        fi

        run_experiment "$2"
        ;;

    "run")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a configuration name${NC}"
            list_configs
            exit 1
        fi

        if [ ! -f "tasks/$2.dstack.yml" ]; then
            echo -e "${YELLOW}Error: Configuration 'tasks/$2.dstack.yml' not found${NC}"
            list_configs
            exit 1
        fi

        echo -e "${BLUE}Applying and running task: $2${NC}"
        dstack apply -f "tasks/$2.dstack.yml"

        # Extract task name from config file
        task_name=$(grep "^name:" "tasks/$2.dstack.yml" | head -1 | awk '{print $2}')

        echo -e "${GREEN}Task applied. To run it:${NC}"
        echo "  dstack run $task_name"
        ;;

    "status")
        echo -e "${GREEN}Current dstack runs:${NC}"
        dstack ps
        ;;

    "logs")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a run name${NC}"
            echo "Use './run_dstack.sh status' to see available runs"
            exit 1
        fi
        dstack logs "$2"
        ;;

    "stop")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a run name${NC}"
            echo "Use './run_dstack.sh status' to see available runs"
            exit 1
        fi
        echo -e "${YELLOW}Stopping run: $2${NC}"
        dstack stop "$2" -y
        ;;

    "delete")
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please specify a run name${NC}"
            echo "Use './run_dstack.sh status' to see available runs"
            exit 1
        fi
        echo -e "${YELLOW}Deleting run: $2${NC}"
        dstack delete "$2" -y
        ;;

    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo "Run './run_dstack.sh' without arguments to see usage"
        exit 1
        ;;
esac
