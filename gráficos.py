# Gera gráficos
total_chamados = serie_temporal_chamados[:, 2]
total_vendas = serie_temporal_vendas[:, 2]
indices = indice[:, 2]
periodo = serie_temporal_vendas[:, [0, 1]]

datas_string = [f"{ano}-{mes:02d}" for ano, mes in periodo]
total_vendas_new = total_vendas/100

# Dados de exemplo
data = {"x": datas_string, "y1": total_vendas_new, "y2": total_chamados, "y3": indices}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
sns.set_theme(style="white", context="talk")

sns.barplot(data=data, x="x", y="y1", color="cyan", ax=ax1)
sns.lineplot(data=data, x="x", y="y2", marker='o', color="gray", ax=ax1, alpha=0.7)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_title("Total de vendas e chamadas por mês", fontsize=15)
ax1.set_ylabel(" ")
ax1.set_ylim(0, np.max(total_vendas_new) * 1.8)

ax2.set_title("Índice de chamadas por total de vendas", fontsize=15)
ax2.axhline(0, color="k", clip_on=False)
sns.barplot(data=data, x="x", y="y3", color="cyan", ax=ax2)
plt.xticks(rotation=45)

sns.despine(bottom=True)
plt.setp(fig.axes, yticks=[])
plt.tight_layout(h_pad=2)

# Exibindo a figura
plt.show()
