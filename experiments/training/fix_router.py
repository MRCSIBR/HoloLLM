with open("chat.py", "r") as f:
    code = f.read()

# Mejorar la regla de detección: si pide explicar, comparar o definir conceptos, va a modo conversación
old_condition = """            if any(k in user_input.lower() for k in ("def ", "code", "function", "python", "script", "program")):
                prompt = f"User: Write a python function for the following task:\\n{user_input}\\n\\nAssistant:\\n```python\\n"
            else:
                prompt = f"User: {user_input}\\n\\nAssistant:\\n" """

new_condition = """            is_explanation = any(k in user_input.lower() for k in ("explain", "what is", "why", "difference", "how", "que es", "explica", "cual es"))
            is_code_request = any(k in user_input.lower() for k in ("write", "create", "implement", "function", "def ", "codigo", "crea", "escribe"))

            if is_code_request and not is_explanation:
                prompt = f"User: Write a python function for the following task:\\n{user_input}\\n\\nAssistant:\\n```python\\n"
            else:
                prompt = f"User: {user_input}\\n\\nAssistant:\\n" """

if "is_explanation =" not in code:
    code = code.replace(
        'if any(k in user_input.lower() for k in ("def ", "code", "function", "python", "script", "program")):\n                prompt = f"User: Write a python function for the following task:\\n{user_input}\\n\\nAssistant:\\n```python\\n"\n            else:\n                prompt = f"User: {user_input}\\n\\nAssistant:\\n"',
        new_condition.strip()
    )
    with open("chat.py", "w") as f:
        f.write(code)
    print("✔ Router de chat actualizado exitosamente.")
else:
    print("El router ya estaba actualizado.")
