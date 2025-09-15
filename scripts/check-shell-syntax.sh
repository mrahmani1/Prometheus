#!/bin/bash

# Simple shell syntax checker
echo "Checking shell script syntax..."

# Check for unmatched quotes
quote_count=$(grep -o "'" scripts/deploy-database-stack.sh | wc -l)
if [ $((quote_count % 2)) -ne 0 ]; then
    echo "ERROR: Unmatched single quotes"
fi

dquote_count=$(grep -o '"' scripts/deploy-database-stack.sh | wc -l)
if [ $((dquote_count % 2)) -ne 0 ]; then
    echo "ERROR: Unmatched double quotes"
fi

# Check for unmatched parentheses
paren_open=$(grep -o '(' scripts/deploy-database-stack.sh | wc -l)
paren_close=$(grep -o ')' scripts/deploy-database-stack.sh | wc -l)
if [ "$paren_open" -ne "$paren_close" ]; then
    echo "ERROR: Unmatched parentheses: $paren_open open, $paren_close close"
fi

# Check for unmatched braces
brace_open=$(grep -o '{' scripts/deploy-database-stack.sh | wc -l)
brace_close=$(grep -o '}' scripts/deploy-database-stack.sh | wc -l)
if [ "$brace_open" -ne "$brace_close" ]; then
    echo "ERROR: Unmatched braces: $brace_open open, $brace_close close"
fi

# Check for unmatched brackets
bracket_open=$(grep -o '\[' scripts/deploy-database-stack.sh | wc -l)
bracket_close=$(grep -o '\]' scripts/deploy-database-stack.sh | wc -l)
if [ "$bracket_open" -ne "$bracket_close" ]; then
    echo "ERROR: Unmatched brackets: $bracket_open open, $bracket_close close"
fi

echo "Basic syntax check completed"