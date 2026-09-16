import ast
import os

def get_imports(path):
    imports = set()
    for root, _, files in os.walk(path):
        if 'venv' in root or '.venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=filepath)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
    return imports

print("\n".join(sorted(get_imports("d:/my project/Project AI Agent/Data_Whisperer_v1.0/backend"))))
