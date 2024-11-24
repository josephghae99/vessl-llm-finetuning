from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import os
from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

# Nom du modèle à utiliser
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
        train_dataset=train_dataset,
        # eval_dataset=val_dataset, # À ajouter si vous avez un ensemble de validation
        tokenizer=tokenizer
    )
    trainer.train()
    # Évaluation (à décommenter si vous avez un ensemble de validation)
    # evaluation_results = trainer.evaluate()
    # print(f"Evaluation results: {evaluation_results}")


# Boucle d'entraînement séquentiel
for i, dataset_path in enumerate(datasets):
    print(f"Début de l'entraînement sur le dataset {i+1}: {dataset_path}")

    if i == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            device_map="auto",
            load_in_4bit=True
        )
        model = prepare_model_for_int8_training(model)
    else:
        print("Chargement du modèle précédent")
        model = AutoModelForCausalLM.from_pretrained(
            f"/artifacts/checkpoint-{i}",
            use_cache=False,
            device_map="auto",
            load_in_4bit=True
        )

    # Configuration de LoRA
    lora_config = LoraConfig(
        r=8,  # Rang de LoRA - à ajuster
        lora_alpha=32,  # Alpha de LoRA - à ajuster
        target_modules=["query_key_value"],  # Modules cibles pour Llama
        lora_dropout=0.05,  # Dropout pour LoRA
        bias="none",  # Type de bias
        task_type="CAUSAL_LM"  # Type de tâche
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Affiche les paramètres entraînables de LoRA

    training_args = TrainingArguments(
        output_dir=f"/artifacts/checkpoint-{i + 1}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # Batch size effective de 8
        num_train_epochs=3,          # Nombre d'epochs - à ajuster
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",       # Sauvegarde à chaque epoch
        learning_rate=2e-5,          # Learning rate - à ajuster
        fp16=True,                   # Active fp16 (à modifier si votre GPU ne le supporte pas)
        # push_to_hub=True,           # Décommenter si vous voulez pousser le modèle sur le Hub
        # hub_model_id="votre_nom_d_utilisateur/nom_du_modele"  # Remplacer par votre nom d'utilisateur et le nom du modèle
    )

    # Charger le dataset (format JSON)
    train_dataset = load_dataset("json", data_files=dataset_path)


    train_and_evaluate(model, training_args, train_dataset["train"])

print("Entraînement séquentiel terminé.")