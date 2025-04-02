local opts = { remap = false, silent = true }

------------------------------------------------------------------------------------
-- Кнопки работы с проектом.
------------------------------------------------------------------------------------

-- Space + A - Куча команд.
tasks = {
	{ "run", [[.venv\Scripts\ruff.exe check src && .venv\Scripts\fastapi.exe dev src\main.py --reload]] },
}

vim.defer_fn(function()
	keys = {}
	for i, value in ipairs(tasks) do
		table.insert(keys, i, value[1])
	end

	vim.keymap.set("n", "<leader>a", function()
		require("fastaction").select(keys, {}, function(item)
			for i, value in ipairs(tasks) do
				if value[1] == item then
					vim.g.hulvdan_task(item, value[2]):start()
				end
			end
		end)
	end, opts)
end, 1)

-- Вставить nocheckin, чтобы лишнего не вкоммичивать.
vim.keymap.set("n", "<C-n>", function()
	vim.api.nvim_input("o# nocheckin<ESC>")
end, opts)

require("conform").setup({
	formatters = {
		black = { command = [[.venv\Scripts\black.exe]] },
		isort = { command = [[.venv\Scripts\isort.exe]] },
	},
})
