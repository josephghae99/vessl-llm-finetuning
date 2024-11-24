from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import os

# Nom du modèle à utiliser (vous l'avez fourni)
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Chemins vers les datasets dans VESSL
datasets = [
    "/dataset/pretraining.jsonl",
    "/dataset/simplified_data_no_rag.jsonl",
    "/dataset/simplified_data_rag.jsonl"
]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def train_and_evaluate(model, training_args, train_dataset):
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset= train_dataset,
     # eval_dataset=val_dataset, #on verra plus tard pour l'évaluation
      tokenizer=tokenizer
  )
  trainer.train()
  #évaluation plus tard
  # evaluation_results = trainer.evaluate()
  # print(f"Evaluation results: {evaluation_results}")


# Boucle d'entraînement séquentiel
for i, dataset_path in enumerate(datasets):
    print(f"Début de l'entraînement sur le dataset {i+1}: {dataset_path}")
    
    if i == 0:
      model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, device_map="auto", load_in_4bit=True)
    else:
      print("chargement du modéle précédent")
      model = AutoModelForCausalLM.from_pretrained(f"/artifacts/checkpoint-{i}", use_cache=False, device_map="auto", load_in_4bit=True)


    training_args = TrainingArguments(
        output_dir=f"/artifacts/checkpoint-{i+1}",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        num_train_epochs=1, #à modifier
        warmup_steps=100,
        logging_steps=10,
        save_strategy = "epoch",
        learning_rate=2e-5, #à modifier
        fp16=True #à modifier si votre GPU ne le permet pas
    )


    # Charger le dataset (on suppose qu'il est au format JSON)
    train_dataset = load_dataset("json", data_files=dataset_path)

    train_and_evaluate(model, training_args, train_dataset["train"])


print("Entraînement séquentiel terminé.")