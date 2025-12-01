#!/bin/bash
# Clear Python cache and verify configuration

echo "========================================="
echo "Clearing Python Cache"
echo "========================================="

# Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Remove all .pyc files
echo "Removing .pyc files..."
find . -name "*.pyc" -exec rm -f {} \; 2>/dev/null

# Verify cache is cleared
echo ""
echo "Checking for remaining cache files..."
REMAINING=$(find . -name "*.pyc" 2>/dev/null | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ All cache files removed!"
else
    echo "⚠ Warning: $REMAINING .pyc files still remain"
fi

echo ""
echo "========================================="
echo "Verifying Configuration"
echo "========================================="

# Check if config.py has correct values
echo "Checking config.py..."
if grep -q "'pipe_pass_bonus': 10," config.py; then
    echo "✓ config.py has correct pipe_pass_bonus value (10)"
else
    echo "✗ ERROR: config.py does not have pipe_pass_bonus: 10"
    grep "pipe_pass_bonus" config.py
fi

echo ""
echo "========================================="
echo "Testing Python Imports"
echo "========================================="

# Test what Python loads
python3 -B -c "
from train_ai import PIPE_PASS_BONUS, DEATH_PENALTY, ALIVE_REWARD
print(f'PIPE_PASS_BONUS  = {PIPE_PASS_BONUS}')
print(f'DEATH_PENALTY    = {DEATH_PENALTY}')
print(f'ALIVE_REWARD     = {ALIVE_REWARD}')

if PIPE_PASS_BONUS == 10:
    print('✓ Correct values loaded!')
else:
    print('✗ ERROR: Wrong values loaded!')
"

echo ""
echo "========================================="
echo "Cache cleared! You can now run:"
echo "  python3 -B test_fixes.py"
echo "  python3 -B train_ai.py"
echo "========================================="

