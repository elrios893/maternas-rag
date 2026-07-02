from src.skills.notifier.skill import NotifierSkill

# Se registra automáticamente al importar el paquete
_notifier_skill = NotifierSkill()
_notifier_skill.register_all()
