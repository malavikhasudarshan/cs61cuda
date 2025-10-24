#!/bin/bash
./cs61cuda > test_output.txt
grep "Validation passed" test_output.txt && echo "✅ Passed auto-validation!" || echo "❌ Validation failed."
